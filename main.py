try:
    import numpy as np
    import cv2
    import os

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
except Exception as e:
    print("Cannot find ",e)
    print("Install Libraries listed in requirement.txt")
    exit()



imageShape = [] #for image shape
#Save the Mask_Result object into a local file

def saveObjects(mask_object, file_name):
    # Unpacking the object
    totalnumber = mask_object.num_obj
    label = mask_object.label_list
    nameLabel = classNames(label)
    score = mask_object.score_list
    boundingbox = mask_object.bounding_box_list
    mask = mask_object.mask_image_list
    # Row
    row = []
    for i in range(len(label)):
        row.append(i)

    #format for saving
    # row label_id score bounding_box masked_img class_name(label_name)
    with open('/Users/lucifer/Desktop/result.txt', 'w') as file:
        file.write("{0}\t{1}\t{2}\t{3}\t\t\t\t\t{4}\t{5}\t\n\n".format("Row","Label_id","Accuracy","Bounding_Box","Mask Images","Class Names"))
        for i in range(len(label)):
            file.write("%i\t" % row[i])
            file.write("%i\t" % label[i])

            file.write("%f\t" % score[i])
            file.write("{}\t".format(boundingbox[i]))
            file.write("{}\t".format(mask[0][i]))
            file.write("%s\t" % nameLabel[i])
            file.write("\n")
    return



#the opposite function of saveOjects() by reading the objects from a file
#file_name: a .txt file that stores the result of mask_object and load the information from it by following the above format
# return an object of result
def loadObjects(file_name):
    #returns a Mask_Result object
    maskResultObj= Mask_Result(len(file_name["instances"].pred_boxes), file_name["instances"].pred_classes, file_name["instances"].scores,
                file_name["instances"].pred_boxes.tensor, file_name["instances"].pred_masks)
    return maskResultObj

#Dictonary to convert class label into names
def classNames(labelList):
    names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",8: "boat", 9: "traffic_light", 10: "fireplug",
        11: "stop_sign", 12: "parking_meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",19: "beef", 20: "elephant",
        21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "bag", 27: "necktie", 28: "bag",29: "frisbee", 30: "ski",
        31: "snowboard", 32: "ball", 33: "kite", 34: "baseball_bat", 35: "baseball_glove", 36: "skateboard",37: "surfboard", 38: "tennis_racket", 39: "bottle", 40: "wineglass",
        41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",49: "orange", 50: "broccoli",
        51: "carrot", 52: "frank", 53: "pizza", 54: "doughnut", 55: "cake", 56: "chair", 57: "sofa", 58: "pot",59: "bed", 60: "dining_table",
        61: "toilet", 62: "television_receiver", 63: "laptop", 64: "mouse", 65: "remote_control",66: "computer_keyboard", 67: "cellular_telephone", 68: "microwave", 69: "oven", 70: "toaster",
        71: "sink", 72: "electric_refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy",78: "hand_blower", 79: "toothbrush"}
    classname = []
    for i in range(len(labelList)):
        a = names[labelList[i]]
        classname.append(a)
    return classname #Return Class names list




#The bounding box class storing the top-left corner and bottom-right corner
class BoundingBox:
    top_left = (0, 0) #the 2D coordinate (x, y) in the image of the top-left corner of the bounding box of a detected object
    bottom_right = (0, 0) #the 2D coordinate (x, y) in the image of the bottom-right corner of the bounding box of a detected object

    def __init__(self, x1, y1, x2, y2):
        BoundingBox.top_left = (x1, y1)
        BoundingBox.bottom_right = (x2, y2)




#The returned result object after processing an image
class Mask_Result:
    num_obj = 0 #the number of detected objects
    label_list = [] #a string list storing the class label of detected objects. The list size should be "num_obj"
    score_list = [] #a float list storeing the  score of each object. The index should be corresponding to the "label_list" index. The list size also should be "num_obj"
    bounding_box_list = [] #the bounding box list with the same index and size as the above two lists
    mask_image_list = [] #The list of mask images with the same index and size as the above lists. This object is a little special as it returns images.


    def __init__(self, number, labels, scores, bounding_boxes, mask_imgs):
        #a simple constructor to assign all the above variables
        Mask_Result.num_obj = number

        #converting all the mask in the image
        masked_list = maskImageGenration(mask_imgs,number)
        Mask_Result.mask_image_list.append(masked_list)

        #  list for class labels
        for i in range(number):
            Mask_Result.label_list.append(int(labels[i]))

        # list for scores
        for i in range(number):
            Mask_Result.score_list.append(float(scores[i]))

        #  Storing Objects for bounding box
        for i in range(number):
            a=bounding_boxes[i]
            x1=float(a[0])
            y1=float(a[1])
            x2=float(a[2])
            y2=float(a[3])
            Mask_Result.bounding_box_list.append(BoundingBox(x1,y1,x2,y2))





#This function is called to train a database by providing a URL. If it is a local file, the URL can be just a local file name
#URL - training data file location (either online or local_file)
#model_file - saved local file model
def training(URL, model_file):

    #train the model and save the model in a local file with the name "model_file"
    return

#This function is called to evaluate the performance of the learned model from a local_file.
#URL - testing data file location (either online or local_file)
#model_file - saved local file model
def evaluate(URL, model_file):

    return


#This function is used to test an input image by outputing a prediction result
#image_file_name: the image used for processing
#return: make sure return an object as a result of the mask RCNN on the image
def predict(image_file_name):
    im = cv2.imread(image_file_name) #Read the file
    #Configuration with detectron2
    global  imageShape
    imageShape = im.shape


    cfg = get_cfg()
    # adding zoo model .
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Adding weights of zoo model from detectron2's
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu' #This line is to used CPU version of detectron2 default mode is GPU
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im) #predicts output
    # Too `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Masked', out.get_image())
    cv2.imwrite(os.path.join('/Users/lucifer/Desktop/Images', "Output.png"), out.get_image())
    print("Hit Esc key start generation of masking images ")
    k = cv2.waitKey(0)
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    print("Waiting for files to be masked ")
    resultObject = loadObjects(outputs)


    return resultObject # Here you should return an result object



# This methord is for converting all the mask into black and white image
obj_imag = []
def maskImageGenration(outputs,length):
    path = '/Users/lucifer/Desktop/Images'
    masked_images = []
    global imageShape
    for c in range(length):
        temp = np.zeros((imageShape[0], imageShape[1]))
        print("",c) #Counter to print which mask is under process
        boolean_value_masked = outputs[c]  #
        for i in range(imageShape[0]):
            for j in range(imageShape[1]):
                if (bool(boolean_value_masked[i][j]) == False):
                    temp[i][j] = 0
                else:
                    temp[i][j] = 255
        masked_images.append(temp)
        cv2.imwrite(os.path.join(path, 'masked{}.png'.format(c)), masked_images[c]) #Saving all the masked images
        obj_imag.append('masked{}.png'.format(c))


    return obj_imag


def main():
    #training("training_url")
    #tesing("testing_url")
    result = predict("/Users/lucifer/Desktop/input2.jpg")

    saveObjects(result, "/Users/lucifer/Desktop/result.txt")

if __name__ == '__main__':
    main()