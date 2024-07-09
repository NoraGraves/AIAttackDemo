# Adapted from https://pyimagesearch.com/2020/10/26/targeted-adversarial-attacks-with-keras-and-tensorflow/

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
import numpy as np
import cv2
import AIAttackDemo.image_processing as image_processing
import AIAttackDemo.neural_net as neural_net
import progressbar

EPS = 2 / 255.0
LR = 5e-3

# clip the values of the tensor to a given range and return it
def clip_eps(tensor, eps):
    return tf.clip_by_value(tensor, clip_value_min=-eps,
        clip_value_max=eps)

# Use gradient method and return targeted delta (noise vector)
def generate_targeted_adversaries(model, baseImage, delta, classIdx,
    target, steps, learning_rate, checkin=False, signs_only=False):
    # initialize optimizer and loss function
    optimizer = Adam(learning_rate=learning_rate)
    sccLoss = SparseCategoricalCrossentropy()
    
    if target == None: untargeted=True
    else: untargeted = False
    
    done_counter = 10

    # iterate over the number of steps
    with progressbar.ProgressBar(max_value=steps) as bar:
        for step in range(0, steps):
            # record our gradients
            with tf.GradientTape() as tape:
                # explicitly indicate that our perturbation vector should
                # be tracked for gradient updates
                tape.watch(delta)
                
                # add our perturbation vector to the base image and
                # preprocess the resulting image
                adversary = tf.clip_by_value((baseImage + delta), clip_value_min=0, clip_value_max=255)
                
                # run this newly constructed image tensor through our
                # model and calculate the loss with respect to the
                # both the *original* class label and the *target*
                # class label
                predictions = model(adversary, training=False)
                
                # find current prediction and change classIdx if necessary
                curr_classIdx = int(np.argmax(predictions.numpy()))
                if (step % 50) == 0:
                    print(f'[CHECK-IN] Current prediction is {neural_net.index2name(curr_classIdx)}')
                    if checkin and (not untargeted):
                        classIdx = curr_classIdx
                
                # update done_counter if current prediction is successful
                if untargeted:
                    if curr_classIdx != classIdx:
                        done_counter = done_counter - 1
                elif curr_classIdx == target:
                        done_counter = done_counter - 1
                        
                # end early if current prediction has been successful ten times
                if done_counter <= 0:
                    print('Goal reached. Ending early.')
                    return delta
                    
                if untargeted:
                    totalLoss = -sccLoss(tf.convert_to_tensor([classIdx]),
                        predictions)
                else:
                    originalLoss = -sccLoss(tf.convert_to_tensor([classIdx]),
                        predictions)
                    targetLoss = sccLoss(tf.convert_to_tensor([target]),
                        predictions)
                    totalLoss = originalLoss + targetLoss
                                            
            # calculate the gradients of loss with respect to the
            # perturbation vector
            gradients = tape.gradient(totalLoss, delta)
            
            # update the weights
            optimizer.apply_gradients([(gradients, delta)])
            
            # use signs only (optional)
            # if signs_only: delta = tf.signs(delta)
            
            # clip perturbation vector and update its value
            delta.assign_add(clip_eps(delta, eps=EPS))
            bar.update(step)

    # return the perturbation vector
    return delta


# Runs a targeted attack on the cropped image and returns the
# adversarial image
def targeted_attack(model, image, target_name, steps=400, eps=EPS, learning_rate=LR, check_in=False):
    target=neural_net.name2index(target_name)
    # Turn image into array
    image = image_processing.preprocess_image(image)

    # create a tensor based off the input image and initialize the
    # perturbation vector (we will update this vector via training)
    baseImage = tf.constant(image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    print("[INFO] finding original classification...")
    # Find current prediction
    orig_class = neural_net.predict_index(image, model)

    print("[INFO] running the attack...")
    # Run the attack
    deltaUpdated = generate_targeted_adversaries(model=model, baseImage=baseImage, delta=delta,
        classIdx=orig_class, target=target, steps=steps, learning_rate=learning_rate, checkin=check_in)

    # create the adversarial example, swap color channels, and save the
    # output image to disk
    print("[INFO] creating targeted adversarial example...")
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    return adverImage

def untargeted_attack(model, image, steps=400, eps=EPS, learning_rate=LR):
    # Turn image into array
    image = image_processing.preprocess_image(image)

    # create a tensor based off the input image and initialize the
    # perturbation vector (we will update this vector via training)
    baseImage = tf.constant(image, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

    print("[INFO] finding original classification...")
    # Find current prediction
    orig_class = neural_net.predict_index(image, model)

    print("[INFO] running the attack...")
    # Run the attack
    deltaUpdated = generate_targeted_adversaries(model=model, baseImage=baseImage, delta=delta,
    classIdx=orig_class, target=None, steps=steps, learning_rate=learning_rate)

    # create the adversarial example, swap color channels, and save the
    # output image to disk
    print("[INFO] creating targeted adversarial example...")
    adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype("uint8")
    adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
    return adverImage

