from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, 'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()

list = detector.detectObjectsFromImage(
    input_image=os.path.join(exec_path , './image/example.jpg'),
    output_image_path=os.path.join(exec_path , 'new_example.jpg'),
    minimum_percentage_probability=80,
    display_percentage_probability=False,
    display_object_name=False
)





