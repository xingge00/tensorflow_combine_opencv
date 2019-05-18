import cv2 as cv


inference_pb = 'F:\Projects\PycharmProjects\opencvtest\detection/test/sorted_inference_graph.pb'
graph_txt = 'F:\Projects\PycharmProjects\opencvtest\detection/test/graph.pbtxt'
net = cv.dnn.readNetFromTensorflow(inference_pb, graph_txt)
layNames = net.getLayerNames()
id = net.getLayerId(layNames[-1])
last_layer = net.getLayer(id)
print(layNames)
print(id)
print(last_layer.type)
image = cv.imread("F:\Projects\PycharmProjects\opencvtest\detection/test_images/5000.jpg")
h, w = image.shape[:2]
cv.imshow("input", image)

im_tensor = cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
net.setInput(im_tensor)
cvOut = net.forward()
print(cvOut.shape)
for detect in cvOut[0,0,:,:]:
    score = detect[2]
    if score > 0.4:
        left = detect[3]*w
        top = detect[4]*h
        right = detect[5]*w
        bottom = detect[6]*h
    cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 255), 4)
cv.imshow("detection_out", image)
cv.waitKey(0)
cv.destroyAllWindows()

