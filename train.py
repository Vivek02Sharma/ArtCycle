import tensorflow as tf
import matplotlib.pyplot as plt

from model import CycleGAN
from loss import Loss_func

# create an optimizer
gen_G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
gen_F_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
disc_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
disc_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

class Train_Model:
    def __init__(self):
        self.gen_G = CycleGAN.build_generator()
        self.gen_F = CycleGAN.build_generator()
        self.disc_X = CycleGAN.build_discriminator()
        self.disc_Y = CycleGAN.build_discriminator()

        self.generator_loss = Loss_func.discriminator_loss
        self.generator_loss = Loss_func.generator_loss
        self.calc_cycle_loss = Loss_func.calc_cycle_loss
        self.identity_loss = Loss_func.identity_loss

    @tf.function
    def train_step(self, real_x, real_y, gen_G, gen_F, disc_X, disc_Y):
        with tf.GradientTape(persistent = True) as tape:
            
            # Forward cycle
            fake_y = gen_G(real_x, training = True)
            cycled_x = gen_F(fake_y, training = True)

            # Backward cycle
            fake_x = gen_F(real_y, training = True)
            cycled_y = gen_G(fake_x, training = True)

            # Identity mapping
            same_y = gen_G(real_y, training = True)
            same_x = gen_F(real_x, training = True)

            # Discriminator outputs
            disc_real_x = disc_X(real_x, training = True)
            disc_real_y = disc_Y(real_y, training = True)
            disc_fake_x = disc_X(fake_x, training = True)
            disc_fake_y = disc_Y(fake_y, training = True)

            # Calculate losses
            gen_G_loss = self.generator_loss(disc_fake_y)
            gen_F_loss = self.generator_loss(disc_fake_x)
            
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
            identity_loss_G = self.identity_loss(real_y, same_y)
            identity_loss_F = self.identity_loss(real_x, same_x)

            total_gen_G_loss = gen_G_loss + total_cycle_loss + identity_loss_G
            total_gen_F_loss = gen_F_loss + total_cycle_loss + identity_loss_F

            disc_X_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Apply gradients
        gen_G_gradients = tape.gradient(total_gen_G_loss, gen_G.trainable_variables)
        gen_F_gradients = tape.gradient(total_gen_F_loss, gen_F.trainable_variables)
        disc_X_gradients = tape.gradient(disc_X_loss, disc_X.trainable_variables)
        disc_Y_gradients = tape.gradient(disc_Y_loss, disc_Y.trainable_variables)

        gen_G_optimizer.apply_gradients(zip(gen_G_gradients, gen_G.trainable_variables))
        gen_F_optimizer.apply_gradients(zip(gen_F_gradients, gen_F.trainable_variables))
        disc_X_optimizer.apply_gradients(zip(disc_X_gradients, disc_X.trainable_variables))
        disc_Y_optimizer.apply_gradients(zip(disc_Y_gradients, disc_Y.trainable_variables))

        return total_gen_G_loss, total_gen_F_loss, disc_X_loss, disc_Y_loss


    def train(self, dataset_X, dataset_Y, epochs):
        for epoch in range(epochs):
            total_gen_G_loss = 0
            total_gen_F_loss = 0
            total_disc_X_loss = 0
            total_disc_Y_loss = 0

            for batch_X, batch_Y in tf.data.Dataset.zip((dataset_X, dataset_Y)):
                gen_G_loss, gen_F_loss, disc_X_loss, disc_Y_loss = self.train_step(batch_X, batch_Y, self.gen_G, self.gen_F, self.disc_X, self.disc_Y)
                
                total_gen_G_loss += gen_G_loss
                total_gen_F_loss += gen_F_loss
                total_disc_X_loss += disc_X_loss
                total_disc_Y_loss += disc_Y_loss

            # Generate image after each epoch
            sample_img = next(iter(dataset_X.take(1)))
            generated = self.gen_G(sample_img)
            plt.figure(figsize = (10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Input Photo")
            plt.imshow(sample_img[0].numpy() * 0.5 + 0.5)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Generated Painting")
            plt.imshow(generated[0].numpy() * 0.5 + 0.5)
            plt.axis('off')
            plt.show()

            print(f'Epoch {epoch + 1}')
            print(f'Generator G Loss: {total_gen_G_loss / len(dataset_X):.4f}')
            print(f'Generator F Loss: {total_gen_F_loss / len(dataset_X):.4f}')
            print(f'Discriminator X Loss: {total_disc_X_loss / len(dataset_X):.4f}')
            print(f'Discriminator Y Loss: {total_disc_Y_loss / len(dataset_X):.4f}\n')

            # Save models every 1 epochs
            if (epoch + 1) % 10 == 0:
                self.gen_G.save(f'gen_G_epoch_{epoch + 1}.h5')
                self.gen_F.save(f'gen_F_epoch_{epoch + 1}.h5')
                self.disc_X.save(f'disc_X_epoch_{epoch + 1}.h5')
                self.disc_Y.save(f'disc_Y_epoch_{epoch + 1}.h5')
    
    # Load the dataset
    def load_data(self, path, batch_size = 32, img_size = (256, 256)):
        dataset = tf.keras.utils.image_dataset_from_directory(
            path,
            label_mode = None,
            image_size = img_size,
            batch_size = batch_size,
        )       
        # Normalize to [-1, 1]
        dataset = dataset.map(lambda x: (x / 127.5) - 1.0)
        return dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

if __name__== "__main___":
    train_model = Train_Model()
    train_photos = train_model.load_data('/kaggle/input/monet2photo/trainB', batch_size = 8)
    train_paintings = train_model.load_data('/kaggle/input/monet2photo/trainA', batch_size = 8)

    train_model.train(train_photos, train_photos, epochs = 100)
    