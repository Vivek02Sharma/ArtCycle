from tensorflow.keras import layers, Model

class CycleGAN:
    def build_generator(self, input_shape = (256, 256, 3)):
        inputs = layers.Input(shape = input_shape)
        
        # Encoder
        x = layers.Conv2D(64, 4, strides = 2, padding = 'same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        # Residual Blocks
        for _ in range(9):
            residual = x
            x = layers.Conv2D(256, 3, padding = 'same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.ReLU()(x)
            
            x = layers.Conv2D(256, 3, padding = 'same')(x)
            x = layers.LayerNormalization()(x)
            x = layers.Add()([x, residual])
            x = layers.ReLU()(x)

        # Decoder
        x = layers.Conv2DTranspose(128, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(64, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2DTranspose(3, 4, strides = 2, padding = 'same', activation = 'tanh')(x)
        
        return Model(inputs, x)

    # Building discriminator
    def build_discriminator(self, input_shape = (256, 256, 3)):
        inputs = layers.Input(shape = input_shape)
        
        x = layers.Conv2D(64, 4, strides = 2, padding = 'same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides = 2, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, 4, strides = 1, padding = 'same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(1, 4, strides = 1, padding = 'same')(x)
        
        return Model(inputs, x)
