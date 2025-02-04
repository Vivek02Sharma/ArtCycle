import tensorflow as tf

class Loss_func:
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    LAMBDA = 10.0 

    def discriminator_loss(self, real, generated):
        real_loss = self.cross_entropy(tf.ones_like(real), real)
        generated_loss = self.cross_entropy(tf.zeros_like(generated), generated)
        return (real_loss + generated_loss) * 0.5

    def generator_loss(self, generated):
        return self.cross_entropy(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real, cycled):
        return tf.reduce_mean(tf.abs(real - cycled)) * self.LAMBDA

    def identity_loss(self, real, same):
        return tf.reduce_mean(tf.abs(real - same)) * self.LAMBDA * 0.5
    
