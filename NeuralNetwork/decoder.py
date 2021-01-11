from keras import layers

def decoder(conv, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, dropout_rate):
    new_conv = conv
    
    for layer in range(convolutional_layers - 1, -1, -1):
        new_conv = layers.Conv2D(convolutional_filters_per_layer[layer], (convolutional_filter_size, convolutional_filter_size), activation='relu', padding='same')(new_conv)
        new_conv = layers.BatchNormalization()(new_conv)
        if layer <= 1:
            new_conv = layers.UpSampling2D((2, 2))(new_conv)
        new_conv = layers.Dropout(dropout_rate)(new_conv)
        
    return layers.Conv2D(1, (convolutional_filter_size, convolutional_filter_size), activation='sigmoid', padding='same')(new_conv)