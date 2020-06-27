"""
Style Transfer wrapped with Runway ML API
"""

import numpy as np
import functools
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import runway
from runway import image


# Load image as an array
def load_img(img):
    max_dim = 512
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
    img = kp_image.img_to_array(img)
  
    img = np.expand_dims(img, axis=0)
    return img
    

def load_and_process_img(img):
    img = load_img(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
  
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

content_layers = ['block5_conv2']
num_content_layers = len(content_layers)

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_style_layers = len(style_layers)

def get_model() :
    # Load the VGG19 model
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build the model 
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target) :
    return tf.reduce_mean(tf.square(base_content - target))


def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
  
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_feature_representations(model, content_img, style_img) :
    # Load images
    content_image = load_and_process_img(content_img)
    style_image = load_and_process_img(style_img)
  
    # Compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)
  
    # Get the style and content feature representations from the model  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


# Computes loss and gradient
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
  
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
  
    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Equally weight each contribution from each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score 
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape : 
        all_loss = compute_loss(**cfg)
        
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


# Performs the style transfer and displays the progress! :)
def run_style_transfer(model, init_image):
    
    num_iterations=800
    content_weight=1e3
    style_weight = 1e-2 
    style_path = "./the-starry-night.jpg"
    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
  
    
    style_img = Image.open(style_path)
    # Get the style and content feature representations (from the article's specified intermediate layers) 
    style_features, content_features = get_feature_representations(model, init_image, style_img)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
    # Set initial image
    init_image = load_and_process_img(init_image)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
  
    # Store best results
    best_loss, best_img = float('inf'), None
  
    loss_weights = (style_weight, content_weight)
    cfg = {'model': model,
           'loss_weights': loss_weights,
           'init_image': init_image,
           'gram_style_features': gram_style_features,
           'content_features': content_features }
    
    # For displaying
    num_rows = 2
    num_cols = 4
    display_interval = num_iterations/(num_rows*num_cols)
  
    # I don't know why these specific values for the mean were chosen either.
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
  
    imgs = []
    for i in range(num_iterations):
        if i % 50 == 0:
            print('iteration %d'%i)

        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
    
        if loss < best_loss:
            # Update best loss and best image from total loss. 
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
      
    return best_img, best_loss


@runway.setup(options={})
def setup():
    return get_model()


@runway.command('convert', inputs={'image': image}, outputs={'image': image})
def sample(model, inputs):
    best_img, best_loss = run_style_transfer(model, inputs['image'])
    return best_img


if __name__ == "__main__":
    runway.run(host='0.0.0.0', port=8000)
