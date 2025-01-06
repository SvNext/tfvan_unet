import tensorflow as tf
from tfvan.encoder import VanTinyEncoder
from tfvan.decoder import VanTinyDecoder

def van_unet_tiny(num_classes):
    decoder_input_shapes = [
        [10, 10, 256], [20, 20, 160],
        [40, 40, 64], [80, 80, 32]
    ]

    embed_dims = (160, 64, 32)
    mlp_ratio = (4, 8, 8)
    depths = (5, 3, 3)


    encoder = VanTinyEncoder()
    decoder = VanTinyDecoder(
        input_shape=decoder_input_shapes[0],
        skip_shapes=decoder_input_shapes[1:],
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratio,
        depths=depths,
    )

    dec_out = decoder(encoder.output)
    output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1),
                                    padding='same', activation='sigmoid')(dec_out)

    return tf.keras.Model(inputs=encoder.input, outputs=output)