
import tensorflow as tf

import tensorflow.keras as keras
import fast_attention
from keras.models import Sequential
from keras.layers import Layer, Conv2D, Conv1D, DepthwiseConv1D, LayerNormalization, Dense, Dropout
from keras.layers import MultiHeadAttention, Embedding, DepthwiseConv2D
from keras.constraints import max_norm



class ScaleNorm(Layer):
	def __init__(self):
		super(ScaleNorm, self).__init__()
		
	@tf.function(jit_compile=False)
	def call(self, x):
		scale = tf.reduce_max(tf.abs(x), axis=-1, keepdims=True)    
		return x / (scale + 1e-8)



class L1Norm(Layer):
	def __init__(self):
		super(L1Norm, self).__init__()
		
	@tf.function(jit_compile=False)
	def call(self, x):
		max_val = tf.reduce_max(x, axis=-1, keepdims=True)
		min_val = tf.reduce_min(x, axis=-1, keepdims=True)
		
		return (x - min_val) / (max_val - min_val + 1e-8)



class TransformerBlock(Layer):
	def __init__(self, embed_dim, num_heads, ff_dim, drop_rate=0.0, fast=True):
		super(TransformerBlock, self).__init__()

		self.num_heads = num_heads
		self.embed_dim = embed_dim
		self.ff_dim = ff_dim
		self.dropout_rate = drop_rate

		if fast:
			self.att = fast_attention.Attention(num_heads=num_heads, hidden_size=embed_dim, attention_dropout=drop_rate)
		else:
			self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=drop_rate)
           
		self.ffn = Sequential(
			[Dense(ff_dim, activation="ReLU", kernel_constraint=max_norm(100.0), bias_constraint=max_norm(100.0)),        # LeakyReLU
			 Dense(embed_dim, kernel_constraint=max_norm(100.0), bias_constraint=max_norm(100.0)),]
		)
		self.norm1 = ScaleNorm()
		self.norm2 = ScaleNorm()
		self.dropout1 = Dropout(drop_rate)
		self.dropout2 = Dropout(drop_rate)

	@tf.function(jit_compile=False, experimental_follow_type_hints=True)
	def call(self, inputs, training=True):
		rescale_coef = 3.0
		attn_output = self.att(inputs, inputs)      # , bias=None
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.norm1(inputs * rescale_coef + attn_output)                    # layernorm
  
		ffn_output = self.ffn(out1)    
		ffn_output = self.dropout2(ffn_output, training=training)
		return self.norm2(inputs * rescale_coef + ffn_output)     

	def get_config(self):
		cfg = super(TransformerBlock, self).get_config()
		cfg.update({'num_heads': self.num_heads,
					'embed_dim': self.embed_dim,
					'ff_dim': self.ff_dim,
					'dropout_rate': self.dropout_rate})
		return cfg
	
	
	

class DenseBlock(Layer):
	def __init__(self, ff_dim, drop_rate=0.1):
		super(DenseBlock, self).__init__()

		self.ff_dim = ff_dim
		self.dropout_rate = drop_rate

		self.norm = ScaleNorm()
		self.dropout = Dropout(drop_rate)
		
		
		
	def build(self, input_shape):
		self.inp_dim = int(input_shape[-1])
		self.ffn = Sequential(
			[Dense(self.ff_dim, activation="ReLU", kernel_constraint=max_norm(100.0), bias_constraint=max_norm(100.0)),        # LeakyReLU
			 Dense(self.inp_dim, kernel_constraint=max_norm(100.0), bias_constraint=max_norm(100.0)),]
		)
		self.norm = ScaleNorm()


	@tf.function(jit_compile=False, experimental_follow_type_hints=True)
	def call(self, inputs, training=True):
		rescale_coef = 3.0
		ffn_output = self.ffn(inputs)    
		ffn_output = self.dropout(ffn_output, training=training)
		return self.norm(inputs * rescale_coef + ffn_output)     

	def get_config(self):
		cfg = super(DenseBlock, self).get_config()
		cfg.update({'inp_dim': self.inp_dim,
					'ff_dim': self.ff_dim,
					'dropout_rate': self.dropout_rate})
		return cfg
	
	
	


class PositionEmbedding(Layer):
	def __init__(self, maxlen, embed_dim):
		super(PositionEmbedding, self).__init__()
		self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
		self.maxlen = maxlen

	@tf.function(jit_compile=False)
	def call(self, x):
		positions = tf.range(start=0, limit=self.maxlen, delta=1)
		positions = self.pos_emb(positions)
		return x + positions

	def get_config(self):
		cfg = super(PositionEmbedding, self).get_config()
		cfg.update({'pos_emb': self.pos_emb,
					'maxlen': self.maxlen})
		return cfg
	



class RandomMask(Layer):
	def __init__(self, maxLen=64, masked_rate=0.75):
		super(RandomMask, self).__init__()

		self.maskedRate = masked_rate
		self.maxLen = maxLen
		self.trainable = False

	@tf.function(jit_compile=False)
	def call(self, inputs, training=None):
		
		batch_size = tf.shape(inputs)[-3]   # or 0 ?
		mask = tf.random.uniform(shape=(batch_size, self.maxLen,), minval=0.0, maxval=1.0, dtype=tf.float32)      # stateless_uniform
		mask = tf.cast(tf.math.greater(mask, self.maskedRate), dtype=tf.float32) 
		
		mask = tf.expand_dims(mask, axis=-1)
		mask = tf.tile(mask, multiples=(1, 1, tf.shape(inputs)[-1]))        # expand the last dimension
		
		if training: 
			return tf.math.multiply(inputs, mask), mask
		
		return inputs, tf.ones(tf.shape(inputs))        # ones = no mask


	def get_config(self):
		cfg = super(RandomMask, self).get_config()
		cfg.update({'maskedRate': self.maskedRate,
					'maxLen': self.maxLen})
		return cfg
	

class RestoreUnmaskedTokens(Layer):
	def __init__(self, maxLen=64):
		super(RestoreUnmaskedTokens, self).__init__()

		self.maxLen = maxLen
		self.trainable = False

	@tf.function(jit_compile=False)
	def call(self, inputs, training=None):
		
		reconstructed, original, mask = inputs
		
		rec = tf.math.multiply(reconstructed, 1.0 - mask)
		remain = tf.math.multiply(original, mask)
		
		if training: 
			return rec + remain
		
		return reconstructed


	def get_config(self):
		cfg = super(RestoreUnmaskedTokens, self).get_config()
		cfg.update({'maxLen': self.maxLen})
		return cfg
	
	

class ConstantLayer(Layer):
	def __init__(self, latent_len = 128, embed_dim = 64, trainable=True, name=None, **kwargs):
		super(ConstantLayer, self).__init__(trainable=trainable, name=name, **kwargs)
		self.latent_len = latent_len
		self.embed_dim = embed_dim


	def build(self, input_shape):
		self.latent_array = self.add_weight(
			shape=(self.latent_len, self.embed_dim),
			initializer="random_normal",
			trainable=True,
		)                           
		
		
	def call(self, input_x):
		bs = tf.shape(input_x)[0]
		x = tf.expand_dims(self.latent_array, axis=0)
		
		return tf.tile(x, multiples=(bs, 1, 1))


	def get_config(self):
		cfg = super(ConstantLayer, self).get_config()
		cfg.update({'latent_len': self.latent_len, 
					'embed_dim': self.embed_dim})
		return cfg
	



class FourierEmbeddingLayer(Layer):
	def __init__(self, embed_dim = 8, trainable=False, name=None, **kwargs):
		super(FourierEmbeddingLayer, self).__init__(trainable=trainable, name=name, **kwargs)
		self.embed_dim = embed_dim // 2

	def build(self, input_shape):
		channel_count = input_shape[-1]
		self.input_len = input_shape[-2]  # sequence length
		
		self.w = tf.range(self.embed_dim, delta=1, dtype=tf.float32)
		self.w = tf.math.pow(2.0, self.w)                         # 1, 2, 4, 8...
		
		self.w = tf.reshape(self.w, [1, 1, 1, -1])       # set batch size to 1, this tensor will be tiled at runtime.    #tf.expand_dims(self.w, axis=0)
		self.w = tf.tile(self.w, multiples=(1, self.input_len, channel_count, 1)) 
			  
		
	@tf.function(jit_compile=False)
	def call(self, seq):
		batch_size = tf.shape(seq)[0]
		channel_count = tf.shape(seq)[-1]

		x = tf.cast(seq * 2.0 * 3.14159265358, dtype=tf.float32)   # 0...2PI

		x = tf.expand_dims(x, axis=-1)
		x = tf.tile(x, multiples=(1, 1, 1, self.embed_dim))  # expand [..., channels, 1] to [..., channels, emb_dim]
		
		w = tf.tile(self.w, multiples=(batch_size, 1, 1, 1)) # adjust for batch size

		seq = tf.concat([tf.math.sin(x * self.w), tf.math.cos(x * self.w)], -1)
		seq = tf.reshape(seq, [batch_size, self.input_len, channel_count*self.embed_dim*2])
		
		return seq


	def get_config(self):
		cfg = super(FourierEmbeddingLayer, self).get_config()
		cfg.update({'input_len': self.input_len, 
					'embed_dim': self.embed_dim})
		return cfg
	



class LinearPositionEmbedding(Layer):
	def __init__(self, maxLen=64):
		super(LinearPositionEmbedding, self).__init__()
		self.maxLen = maxLen

	@tf.function(jit_compile=False)
	def call(self, x):
		
		batch_size = tf.shape(x)[0]
		
		pos_emb = tf.cast(tf.range(self.maxLen) / self.maxLen, tf.float32)
		pos_emb = tf.reshape(pos_emb, [1, self.maxLen, 1])

		pos_emb = tf.tile(pos_emb, multiples=(batch_size, 1, 1)) # adjust for batch size

		return tf.concat([x, pos_emb], -1)    

	def get_config(self):
		cfg = super(LinearPositionEmbedding, self).get_config()
		cfg.update({'maxLen': self.maxLen})
		return cfg
	


class AugmentAmplitude(Layer):
	def __init__(self, mean_aug=0.1, percent_aug=0.1):
		super(AugmentAmplitude, self).__init__()
		self.mean_aug = mean_aug
		self.percent_aug = percent_aug
		self.trainable = False

	@tf.function(jit_compile=False)
	def call(self, x, training=None):

		inp_shape = tf.shape(x)
		batch_size = inp_shape[0]
		
		if not training:
			return x
		
		mean_shift = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32) * self.mean_aug    
		# for mean_aug=0.1   -   (0.0, 0.1)
		
		relative_change = tf.random.uniform([batch_size], minval=1.0-self.percent_aug, maxval=1.0+self.percent_aug, dtype=tf.float32)  
		# for 0.1 relative augmentation, get random on (0.9, 1.1)


		relative_change = tf.reshape(relative_change, shape=[batch_size, 1])
		mean_shift = tf.reshape(mean_shift, shape=[batch_size, 1])
		
		relative_change = tf.tile(relative_change, multiples=(1, inp_shape[-1]))        # expand the last dimension
		mean_shift = tf.tile(mean_shift, multiples=(1, inp_shape[-1])) 
		
		
		return x * relative_change + mean_shift 


	def get_config(self):
		cfg = super(AugmentAmplitude, self).get_config()
		cfg.update({'mean_aug': self.mean_aug, 
					'percent_aug': self.percent_aug})
		return cfg
	
 
 
	
 
class UNetBlock2D(Layer):
    def __init__(self, filters, trainable=True,
                name=None, **kwargs):
        super(UNetBlock2D, self).__init__(trainable=trainable, name=name, **kwargs)
        self.filters = filters
        #self.act = tf.nn.leaky_relu

    def build(self, input_shape):
                    
        
        self.residual = Conv2D(filters=int(self.filters), kernel_size=1, use_bias=True, 
                                    kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
        
        self.conv1 = Conv2D(filters=self.filters, kernel_size=3, use_bias=True, padding="SAME",
                                    kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
        
        self.conv2 = Conv2D(filters=self.filters, kernel_size=3, use_bias=True, padding="SAME",
                                    kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
        

        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, input_x):

        x = self.norm1(input_x)
        x = tf.nn.leaky_relu(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        
        if input_x.shape[1:-1] == x.shape[1:-1]:
            x += self.residual(input_x)
        
        return x


    def get_config(self):
        cfg = super(UNetBlock2D, self).get_config()
        cfg.update({'filters': self.filters})
        return cfg
    
    
    
    

class InvertedResidual1D(Layer):
	def __init__(self, filters, strides, expansion_factor=2, trainable=True,
				name=None, **kwargs):
		super(InvertedResidual1D, self).__init__(trainable=trainable, name=name, **kwargs)
		self.filters = filters
		self.strides = strides
		self.expansion_factor = expansion_factor	# allowed to be decimal value
		self.act = tf.nn.leaky_relu  #tf.nn.leaky_relu

	def build(self, input_shape):
		input_channels = int(input_shape[-1])
		
		l2_reg = 0.0
		
		self.ptwise_conv1 = Conv1D(filters=int(input_channels*self.expansion_factor), kernel_size=1, use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
	
		self.dwise = DepthwiseConv1D(kernel_size=3, strides=self.strides, padding='same', use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
		
		self.ptwise_conv2 = Conv1D(filters=self.filters, kernel_size=1, use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))

		self.bn1 = ScaleNorm()
		self.bn2 = ScaleNorm()
		
	@tf.function(jit_compile=False)
	def call(self, input_x):

		x = self.ptwise_conv1(input_x)
		x = self.bn1(x)
		x = self.act(x)

		x = self.dwise(x)
		x = self.bn2(x)
		x = self.act(x)

		x = self.ptwise_conv2(x)


		if input_x.shape[1:] == x.shape[1:]:
			x += input_x
		return x

	def get_config(self):
		cfg = super(InvertedResidual1D, self).get_config()
		cfg.update({'filters': self.filters,
					'strides': self.strides,
					'expansion_factor': self.expansion_factor})
		return cfg
	
	
 
 
 

class InvertedResidual2D(Layer):
	def __init__(self, filters, strides, expansion_factor=2, trainable=True,
				name=None, **kwargs):
		super(InvertedResidual2D, self).__init__(trainable=trainable, name=name, **kwargs)
		self.filters = filters
		self.strides = strides
		self.expansion_factor = expansion_factor	# allowed to be decimal value
		self.act = tf.nn.leaky_relu  #tf.nn.leaky_relu

	def build(self, input_shape):
		input_channels = int(input_shape[-1])
		
		self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor), kernel_size=1, use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
	
		self.dwise = DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))
		
		self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1, use_bias=True, 
									kernel_constraint=max_norm(6.0), bias_constraint=max_norm(6.0))

		self.bn1 = ScaleNorm()
		self.bn2 = ScaleNorm()
		
	@tf.function(jit_compile=False)
	def call(self, input_x):

		x = self.ptwise_conv1(input_x)
		x = self.bn1(x)
		x = self.act(x)

		x = self.dwise(x)
		x = self.bn2(x)
		x = self.act(x)

		x = self.ptwise_conv2(x)


		if input_x.shape[1:] == x.shape[1:]:
			x += input_x
		return x

	def get_config(self):
		cfg = super(InvertedResidual2D, self).get_config()
		cfg.update({'filters': self.filters,
					'strides': self.strides,
					'expansion_factor': self.expansion_factor})
		return cfg
	
	