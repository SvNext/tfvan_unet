import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from tfvan.block import Block
from tfvan.embed import PatchEmbedding
from tfvan.norm import LayerNorm


def VanTinyDecoder(
        input_shape, skip_shapes, embed_dims, mlp_ratios, depths,
        drop_rate=0., path_drop=0.1, model_name='decoder',
        ):

    x_input = layers.Input(shape=input_shape)
    skips = [layers.Input(shape=z) for z in skip_shapes]

    z = x_input
    for i, (skip, embed_dim, block_depths, mlp_ratio) in enumerate(zip(skips, embed_dims, depths, mlp_ratios), start=1):
        z = layers.Conv2DTranspose(z.shape[-1], kernel_size=(3, 3), strides=(2, 2), padding='same')(z)
        z = layers.Concatenate()([skip, z])

        for j in range(block_depths):
            block_name = f'block_dec{i}.{j}'
            z = Block(
                mlp_ratio=mlp_ratio, mlp_drop=drop_rate, path_drop=path_drop, name=block_name)(z)

        z = LayerNorm(name=f'norm_dec{i}')(z)
        if i != len(depths):
            z = layers.Conv2D(embed_dim, kernel_size=(1, 1), padding='same')(z)

    return models.Model(inputs=[x_input] + skips, outputs=z, name=model_name)



if __name__ == '__main__':

    from training.court.van_encoder import VanTinyEncoder

    input_shape = (320, 320, 3)
    embed_dims = (160, 64, 32)
    mlp_ratio = (4, 8, 8)
    depths = (5, 3, 3)

    x = np.random.uniform(0, 255, (1,) + input_shape)
    encoder = VanTinyEncoder()
    decoder_input_shapes = [
        [10, 10, 256], [20, 20, 160],
        [40, 40, 64], [80, 80, 32]
    ]

    decoder = VanTinyDecoder(
        input_shape=decoder_input_shapes[0],
        skip_shapes=decoder_input_shapes[1:],
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratio,
        depths=depths,
    )

    enc_out = encoder(x)
    dec_out = decoder(enc_out)
    #
    print(dec_out.shape)


    for yy in enc_out:
        print(yy.shape)
