import numpy as np
import tensorflow as tf
import download
#from cache import cache
import os
import sys
from glob import glob
from PIL import Image
import csv
import pandas as pd
# Internet URL for the tar-file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
# Directory to store the downloaded data.
data_dir = "inception/"
# File containing the mappings between class-number and uid. (Downloaded)
path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"
# File containing the mappings between uid and string. (Downloaded)
path_uid_to_name = "imagenet_synset_to_human_label_map.txt"
# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "classify_image_graph_def.pb"

'''
1. THIS IS A CODE INSPIRED FROM inception.py FROM IMAGENET 
2. HERE THE CODE IS ONLY USED TO GENERATE TRANSFER VALUES ID BOTTLENECK VALUES
3. THE WEIGHTS FOR THE BOTTLE NECK LAYER I.E LAST TWO LAYERS (FULL CONNECTED SOFTMAX LAYER AND CLASSIFICATION LAYER IS REMOVED
4. THE WEIGHTS ARE SAVED IN A CSV FILE 
5. THESE WEIGHTS WILL FURTHER BE USED FOR TRAINING AND CLASSIFICATION
6. REMEMBER THAT THE MODEL IS NOT TRAIN AGAIN ON THE TILES RATHER THE PRE-TRAINED MODEL IS USED TO GENERATE TRANSFER VALUES

'''

def maybe_download():
    """
    Download the Inception model from the internet if it does not already
    exist in the data_dir. The file is about 85 MB.
    """

    print("Downloading Inception v3 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


class NameLookup:
    """
    Used for looking up the name associated with a class-number.
    This is used to print the name of a class instead of its number,
    e.g. "plant" or "horse".

    Maps between:
    - cls is the class-number as an integer between 1 and 1000 (inclusive).
    - uid is a class-id as a string from the ImageNet data-set, e.g. "n00017222".
    - name is the class-name as a string, e.g. "plant, flora, plant life"
   """

    def __init__(self):
        # Mappings between uid, cls and name are dicts, where insertions and
        # lookup have O(1) time-usage on average, but may be O(n) in worst case.
        self._uid_to_cls = {}   # Map from uid to cls.
        self._uid_to_name = {}  # Map from uid to name.
        self._cls_to_uid = {}   # Map from cls to uid.

        # Read the uid-to-name mappings from file.
        path = os.path.join(data_dir, path_uid_to_name)
        with open(file=path, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # Remove newlines.
                line = line.replace("\n", "")

                # Split the line on tabs.
                elements = line.split("\t")

                # Get the uid.
                uid = elements[0]

                # Get the class-name.
                name = elements[1]

                # Insert into the lookup-dict.
                self._uid_to_name[uid] = name

        # Read the uid-to-cls mappings from file.
        path = os.path.join(data_dir, path_uid_to_cls)
        with open(file=path, mode='r') as file:
            # Read all lines from the file.
            lines = file.readlines()

            for line in lines:
                # We assume the file is in the proper format,
                # so the following lines come in pairs. Other lines are ignored.

                if line.startswith("  target_class: "):
                    # This line must be the class-number as an integer.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the class-number as an integer.
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    # This line must be the uid as a string.

                    # Split the line.
                    elements = line.split(": ")

                    # Get the uid as a string e.g. "n01494475"
                    uid = elements[1]

                    # Remove the enclosing "" from the string.
                    uid = uid[1:-2]

                    # Insert into the lookup-dicts for both ways between uid and cls.
                    self._uid_to_cls[uid] = cls
                    self._cls_to_uid[cls] = uid

    def uid_to_cls(self, uid):
        """
        Return the class-number as an integer for the given uid-string.
        """

        return self._uid_to_cls[uid]

    def uid_to_name(self, uid, only_first_name=False):
        """
        Return the class-name for the given uid string.

        Some class-names are lists of names, if you only want the first name,
        then set only_first_name=True.
        """

        # Lookup the name from the uid.
        name = self._uid_to_name[uid]

        # Only use the first name in the list?
        if only_first_name:
            name = name.split(",")[0]

        return name

    def cls_to_name(self, cls, only_first_name=False):
        """
        Return the class-name from the integer class-number.

        Some class-names are lists of names, if you only want the first name,
        then set only_first_name=True.
        """

        # Lookup the uid from the cls.
        uid = self._cls_to_uid[cls]

        # Lookup the name from the uid.
        name = self.uid_to_name(uid=uid, only_first_name=only_first_name)

        return name


########################################################################


class Inception:
    """
    The Inception model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the Inception model
    will be loaded and can be used immediately without training.

    The Inception model can also be used for Transfer Learning.
    """

    # Name of the tensor for feeding the input image as jpeg.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"

    # Name of the tensor for feeding the decoded input image.
    # Use this for feeding images in other formats than jpeg.
    tensor_name_input_image = "DecodeJpeg:0"

    # Name of the tensor for the resized input image.
    # This is used to retrieve the image after it has been resized.
    tensor_name_resized_image = "ResizeBilinear:0"

    # Name of the tensor for the output of the softmax-classifier.
    # This is used for classifying images with the Inception model.
    tensor_name_softmax = "softmax:0"

    # Name of the tensor for the unscaled outputs of the softmax-classifier (aka. logits).
    tensor_name_softmax_logits = "softmax/logits:0"

    # Name of the tensor for the output of the Inception model.
    # This is used for Transfer Learning.
    tensor_name_transfer_layer = "pool_3:0"

    def __init__(self):
        # Mappings between class-numbers and class-names.
        # Used to print the class-name as a string e.g. "horse" or "plant".
        self.name_lookup = NameLookup()

        # Now load the Inception model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the Inception model from the proto-buf file.

        # Get the output of the Inception model by looking up the tensor
        # with the appropriate name for the output of the softmax-classifier.
        self.y_pred = self.graph.get_tensor_by_name(self.tensor_name_softmax)

        # Get the unscaled outputs for the Inception model (aka. softmax-logits).
        self.y_logits = self.graph.get_tensor_by_name(self.tensor_name_softmax_logits)

        # Get the tensor for the resized image that is input to the neural network.
        self.resized_image = self.graph.get_tensor_by_name(self.tensor_name_resized_image)

        # Get the tensor for the last layer of the graph, aka. the transfer-layer.
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

        # Get the number of elements in the transfer-layer.
        self.transfer_len = self.transfer_layer.get_shape()[3]

        # Create a TensorFlow session for executing the graph.
        self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        Call this function when you are done using the Inception model.
        It closes the TensorFlow session to release its resources.
        """

        self.session.close()

    def _write_summary(self, logdir='summary/'):
        """
        Write graph to summary-file so it can be shown in TensorBoard.

        This function is used for debugging and may be changed or removed in the future.

        :param logdir:
            Directory for writing the summary-files.

        :return:
            Nothing.
        """

        writer = tf.train.SummaryWriter(logdir=logdir, graph=self.graph)
        writer.close()

    def _create_feed_dict(self, image_path=None, image=None):
        """
        Create and return a feed-dict with an image.

        :param image_path:
            The input image is a jpeg-file with this file-path.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the Inception graph in TensorFlow.
        """

        if image is not None:
            # Image is passed in as a 3-dim array that is already decoded.
            feed_dict = {self.tensor_name_input_image: image}

        elif image_path is not None:
            # Read the jpeg-image as an array of bytes.
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            # Image is passed in as a jpeg-encoded image.
            feed_dict = {self.tensor_name_input_jpeg: image_data}

        else:
            raise ValueError("Either image or image_path must be set.")

        return feed_dict

    def classify(self, image_path=None, image=None):
        """
        Use the Inception model to classify a single image.

        The image will be resized automatically to 299 x 299 pixels,
        see the discussion in the Python Notebook for Tutorial #07.

        :param image_path:
            The input image is a jpeg-file with this file-path.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Array of floats (aka. softmax-array) indicating how likely
            the Inception model thinks the image is of each given class.
        """

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # Execute the TensorFlow session to get the predicted labels.
        pred = self.session.run(self.y_pred, feed_dict=feed_dict)

        # Reduce the array to a single dimension.
        pred = np.squeeze(pred)

        return pred

    def get_resized_image(self, image_path=None, image=None):
        """
        Input an image to the Inception model and return
        the resized image. The resized image can be plotted so
        we can see what the neural network sees as its input.

        :param image_path:
            The input image is a jpeg-file with this file-path.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            A 3-dim array holding the image.
        """

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # Execute the TensorFlow session to get the predicted labels.
        resized_image = self.session.run(self.resized_image, feed_dict=feed_dict)

        # Remove the 1st dimension of the 4-dim tensor.
        resized_image = resized_image.squeeze(axis=0)

        # Scale pixels to be between 0.0 and 1.0
        resized_image = resized_image.astype(float) / 255.0

        return resized_image

    def print_scores(self, pred, k=10, only_first_name=True):
        """
        Print the scores (or probabilities) for the top-k predicted classes.

        :param pred:
            Predicted class-labels returned from the predict() function.

        :param k:
            How many classes to print.

        :param only_first_name:
            Some class-names are lists of names, if you only want the first name,
            then set only_first_name=True.

        :return:
            Nothing.
        """

        # Get a sorted index for the pred-array.
        idx = pred.argsort()

        # The index is sorted lowest-to-highest values. Take the last k.
        top_k = idx[-k:]

        # Iterate the top-k classes in reversed order (i.e. highest first).
        for cls in reversed(top_k):
            # Lookup the class-name.
            name = self.name_lookup.cls_to_name(cls=cls, only_first_name=only_first_name)

            # Predicted score (or probability) for this class.
            score = pred[cls]

            # Print the score and class-name.
            print("{0:>6.2%} : {1}".format(score, name))

    def transfer_values(self, image_path=None, image=None):
        """
        Calculate the transfer-values for the given image.
        These are the values of the last layer of the Inception model before
        the softmax-layer, when inputting the image to the Inception model.

        The transfer-values allow us to use the Inception model in so-called
        Transfer Learning for other data-sets and different classifications.

        It may take several hours or more to calculate the transfer-values
        for all images in a data-set. It is therefore useful to cache the
        results using the function transfer_values_cache() below.

        :param image_path:
            The input image is a jpeg-file with this file-path.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            The transfer-values for those images.
        """

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        # Use TensorFlow to run the graph for the Inception model.
        # This calculates the values for the last layer of the Inception model
        # prior to the softmax-classification, which we call transfer-values.
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # Reduce to a 1-dim array.
        transfer_values = np.squeeze(transfer_values)

        return transfer_values


########################################################################
# Batch-processing.


def process_images(fn, images=None, image_paths=None):
    """
    Call the function fn() for each image, e.g. transfer_values() from
    the Inception model above. All the results are concatenated and returned.

    :param fn:
        Function to be called for each image.

    :param images:
        List of images to process.

    :param image_paths:
        List of file-paths for the images to process.

    :return:
        Numpy array with the results.
    """

    # Are we using images or image_paths?
    using_images = images is not None

    # Number of images.
    if using_images:
        num_images = len(images)
    else:
        num_images = len(image_paths)

    # Pre-allocate list for the results.
    # This holds references to other arrays. Initially the references are None.
    result = [None] * num_images

    # For each input image.
    for i in range(num_images):
        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Process the image and store the result for later use.
        if using_images:
            result[i] = fn(image=images[i])
        else:
            result[i] = fn(image_path=image_paths[i])

    # Print newline.
    print()

    # Convert the result to a numpy array.
    result = np.array(result)

    return result


if __name__ == '__main__':
    print(tf.__version__)

    # Download Inception model if not already done.
    maybe_download()

    # Load the Inception model so it is ready for classifying images.
    model = Inception()
    for j in range(262):  # specify the number of directories which contain images
        image_path ='/ysm-gpfs/pi/gerstein/aj557/data_deeppath/data_slices/dir_{}/tile_out*/label*/*jpeg'.format(i+1)
        images=glob(image_path) # list all image paths in a particular directory
        labels=[]  #list to store the class or label of each tile in the slide 
        images_jpeg=[]
        transfer_values_train=[]
        image_name=[]
    # print(image)
        for i in range(len(images)):
            image_name.append(os.path.basename(images[i]))#'im{}'.format(i+1)
            labels.append((os.path.split(os.path.split(images[i])[0])[1])[6])
            name=Image.open(images[i])
            images_jpeg.append(name)
            transfer_values_train.append(model.transfer_values(image_path=images[i],image=images_jpeg[i]))
            name.close()
        #transfer_values.append([identity,transfer_values_train.tolist()])
        #print(transfer_values)
        transfer_values=pd.DataFrame(data=transfer_values_train)   # 2048 transfer values are generated for each tile( data frame size is (n_tiles,2048))
        #image_name=pd.DataFrame(data=image_name)
	# ADD THE INDETITY AND LABEL COLUMNS TO THE TRANSFER VALUES DATA FRAME ; SIZE NOW IS N_IMAGES*2050
        transfer_values['image_name']=image_name
        transfer_values['label']=labels
        #print(transfer_values)
        with open('/ysm-gpfs/pi/gerstein/aj557/data_deeppath/transfer_values{}.csv'.format(i+1),'a') as f:  #creates a csv file and appends the transfer values and label for each tile
            transfer_values.to_csv(f,sep=',')
    
    model.close()

