def get_user_answer_boolean(prompt):
    print()
    user_answer = input(prompt).upper()
    
    while user_answer != 'Y' and user_answer != 'N':
        print("Invalid input!")
        user_answer = input(prompt)
        
    return user_answer == 'Y'

def init_gpu():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)