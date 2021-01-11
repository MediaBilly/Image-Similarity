from keras import layers


def encoder_layers(convolutional_layers):
    if convolutional_layers <= 2:
        return convolutional_layers * 4
    
    return 6 + (convolutional_layers - 2) * 3



def encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, dropout_rate):
    conv = input_img

    for layer in range(convolutional_layers):
        conv = layers.Conv2D(convolutional_filters_per_layer[layer], (convolutional_filter_size, convolutional_filter_size), activation='relu', padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        if layer <= 1:
            conv = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        conv = layers.Dropout(dropout_rate)(conv)
        
    return conv
