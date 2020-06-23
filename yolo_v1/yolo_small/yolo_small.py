import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
gpu = tf.config.experimental.get_visible_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)
import tensorflow.keras.layers as layers
import cv2
import numpy as np
import imageio


def leak_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)


def relu(x):
    return tf.maximum(x, 0)


class YoloInf():
    def __init__(self, cpkt_file, B=2, S=7, name=None, verbose=False, nms='nms'):
        self.B = B
        self.S = S
        self.name = name
        self.weight_file = cpkt_file
        self.verbose = verbose
        self.classes = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.C = len(self.classes)
        self.iou_threshold = 0.4  # iou scores threshold
        self.confidence_threshold = 0.1  # confidence scores threhold
        self.max_output_size = self.S * self.S * self.B
        self.nms_method = nms
        self.score_proposal = 0.5

        self._build_net()
        self._load_weight()

    def nms(self, bboxes, scores, classes):
        """
        input:
            bboxes (tensor) with shape (N, 4)
            scores (tensor) with shape (N, )
            classes (tensor) with shape (N, )
        output:
            boxes, scores and classes after non maximum supression with shape (N, 4), (N, ) and (N,) respectively
        """

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        boxes_area = (x2 - x1) * (y2 - y1)
        num = bboxes.shape[0]
        B = tf.range(num)
        res = []
        while B.shape[0] > 0:
            bm = tf.gather(B, tf.math.argmax(tf.gather(scores, B)))
            res.append(bm.numpy())
            if B.shape[0] == 1 or len(res) >= self.max_output_size:
                break
            box_m = tf.gather(bboxes, bm)
            boxes_remain = tf.gather(bboxes, B)
            # calculate the intersection area of the selected box and others
            inter_x1 = tf.maximum(box_m[0], boxes_remain[:, 0])
            inter_x2 = tf.minimum(box_m[2], boxes_remain[:, 2])
            inter_y1 = tf.maximum(box_m[1], boxes_remain[:, 1])
            inter_y2 = tf.minimum(box_m[3], boxes_remain[:, 3])
            intersect_area = tf.maximum(0, (inter_x2 - inter_x1)) * tf.maximum(0, (inter_y2 - inter_y1))
            union_area = tf.gather(boxes_area, bm) + tf.gather(boxes_area, B) - intersect_area
            iou = intersect_area / union_area
            filter_mask = iou <= self.iou_threshold
            B = tf.boolean_mask(B, filter_mask)
        return tf.gather(bboxes, res), tf.gather(scores, res), tf.gather(classes, res)

    def soft_nms(self, bboxes, scores, classes):
        """
        input:
            bboxes (tensor) with shape (N, 4)
            scores (tensor) with shape (N, )
        output:
            filtered boxes after soft non maximum supression with shape (N, 4) and corresponding score with shape (N,)
        """
        classes = tf.cast(classes, tf.float32)
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        boxes_area = (x2 - x1) * (y2 - y1)
        _scores = tf.Variable(scores)
        _bboxes = tf.Variable(bboxes)
        _classes = tf.Variable(classes)
        res_bboxes = tf.random.normal(shape=(1, 4))
        res_scores = tf.random.normal(shape=(1,))
        res_classes = tf.random.normal(shape=(1,))
        while _bboxes.shape[0] > 0 and res_bboxes.shape[0] + 1 <= self.max_output_size:
            bm = tf.argmax(_scores).numpy()
            res_bboxes = tf.concat((res_bboxes, tf.reshape(tf.gather(_bboxes, bm), (-1, 4))), axis=0)
            res_scores = tf.concat((res_scores, tf.reshape(tf.gather(_scores, bm), (-1,))), axis=0)
            res_classes = tf.concat((res_classes, tf.reshape(tf.gather(_classes, bm), (-1,))), axis=0)
            num = _bboxes.shape[0]
            if num == 1:
                break
            box_m = tf.gather(_bboxes, bm)
            area_m = tf.gather(boxes_area, bm)
            indices = [i for i in range(num) if i != bm]
            _bboxes = tf.gather(_bboxes, indices)
            _scores = tf.gather(_scores, indices)
            _classes = tf.gather(_classes, indices)
            boxes_area = tf.gather(boxes_area, indices)
            # calculate the intersection area of the selected box and others
            inter_x1 = tf.maximum(box_m[0], _bboxes[:, 0])
            inter_x2 = tf.minimum(box_m[2], _bboxes[:, 2])
            inter_y1 = tf.maximum(box_m[1], _bboxes[:, 1])
            inter_y2 = tf.minimum(box_m[3], _bboxes[:, 3])
            intersect_area = tf.maximum(0, inter_x2 - inter_x1) * tf.maximum(0, inter_y2 - inter_y1)
            union_area = area_m + boxes_area - intersect_area
            iou = intersect_area / union_area
            # updates the scores
            _scores = tf.where(iou <= self.iou_threshold, _scores, self.score_proposal * _scores)
        return res_bboxes[1:], res_scores[1:], tf.cast(res_classes[1:], tf.int32)

    def _build_net(self, input_shape=(448, 448, 3)):
        print('build neural network..')
        # block 1
        convolution_act = leak_relu
        dense_act = leak_relu
        input = layers.Input(shape=input_shape)
        conv1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
                              padding='same', activation=convolution_act)(input)
        max_pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool1")(conv1)

        # block 2
        conv2 = layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), name="conv2",
                              padding='same', activation=convolution_act)(max_pool1)
        max_pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool2")(conv2)
        # block 3
        conv3 = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), name="conv3",
                              padding='same', activation=convolution_act)(max_pool2)
        conv4 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), name="conv4",
                              padding='same', activation=convolution_act)(conv3)
        conv5 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), name="conv5",
                              padding='same', activation=convolution_act)(conv4)
        conv6 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name="conv6",
                              padding='same', activation=convolution_act)(conv5)
        max_pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="max_pool3")(conv6)

        # block 4
        conv7 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), name="conv7",
                              padding='same', activation=convolution_act)(max_pool3)
        conv8 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name="conv8",
                              padding='same', activation=convolution_act)(conv7)
        conv9 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), name="conv9",
                              padding='same', activation=convolution_act)(conv8)
        conv10 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name="conv10",
                               padding='same', activation=convolution_act)(conv9)
        conv11 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), name="conv11",
                               padding='same', activation=convolution_act)(conv10)
        conv12 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name="conv12",
                               padding='same', activation=convolution_act)(conv11)
        conv13 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), name="conv13",
                               padding='same', activation=convolution_act)(conv12)
        conv14 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), name="conv14",
                               padding='same', activation=convolution_act)(conv13)
        conv15 = layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv15',
                               padding='same', activation=convolution_act)(conv14)
        conv16 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='conv16',
                               padding='same', activation=convolution_act)(conv15)

        max_pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool4')(conv16)

        # block 5print(proba_filtered.shape, boxes_filtered.shape, class_filtered.shape)
        conv17 = layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv17',
                               padding='same', activation=convolution_act)(max_pool4)
        conv18 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='conv18',
                               padding='same', activation=convolution_act)(conv17)
        conv19 = layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), name='conv19',
                               padding='same', activation=convolution_act)(conv18)
        conv20 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='conv20',
                               padding='same', activation=convolution_act)(conv19)
        conv21 = layers.Conv2D(1024, kernel_size=(3, 3,), strides=(1, 1), name='conv21',
                               padding='same', activation=convolution_act)(conv20)
        conv22 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), name='conv22',
                               padding='same', activation=convolution_act)(conv21)

        # block 6
        conv23 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='conv23',
                               padding='same', activation=convolution_act)(conv22)
        conv24 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), name='conv24',
                               padding='same', activation=convolution_act)(conv23)
        conv24 = tf.transpose(conv24, (0, 3, 1, 2))
        flatten = layers.Flatten()(conv24)
        dense1 = layers.Dense(512, activation=dense_act)(flatten)
        dense2 = layers.Dense(4096, activation=dense_act)(dense1)
        dense3 = layers.Dense(self.S * self.S * (self.C + 5 * self.B), activation=None)(dense2)
        self.model = tf.keras.Model(inputs=input, outputs=dense3)
        if self.verbose:
            print(self.model.summary())

    def _preprocess_ckpt(self):
        pass
        # ckpt = self.weight_file
        # ckpt = './data/v12/YOLO_small.ckpt'
        # ckpt_reader = tf.train.load_checkpoint(ckpt)
        # weights_list = []
        # weights_list.append(ckpt_reader.get_tensor('Variable'))
        # for i in range(1, 54):
        #     weight = ckpt_reader.get_tensor('Variable_' + str(i))
        #     weights_list.append(weight)
        # return weights_list

    def _load_weight(self):
        pass
        # weight_list = self._preprocess_ckpt()
        # # print(len(weight_list))
        # self.model.set_weights(weight_list)
        # self.model.save_weights(self.weight_file)
        # print('set weights finished')
        # weight_path = './weights/YOLO_small_v1.ckpt'
        # self.model.load_weights(weight_path)
        # print('loading weight finished')

    def forward(self, img):
        # forward the image using neural network constructed
        if isinstance(img, str):
            img = np.asarray(imageio.imread(img))
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise ValueError('img should be a str representing the file path of an image or a numpy.ndarray')
        return self.model(img)

    def _build_detector(self, predict, img_shape=(448, 448, 3)):
        # interpret the net output and calculate the predicting boxes for visualization
        # output tensor with shape (None, self.S * self.S * (self.C + 5 * self.B)), i.e 7*7*30
        # in each grid cell , (none, :2), (none, 2:10), (none, 10:) are representing Pr(object),
        # boxes coordinates (x, y, w, h) and Pr(classes|object) respectively
        x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)] *
                                                    self.B * self.S), [self.B, self.S, self.S]), (1, 2, 0))
        y_offset = np.transpose(x_offset, (1, 0, 2))
        # reshape predict (none, 1470) -> (none, 7, 7, 30)
        # predict = tf.random.uniform(shape=(1, 1470), minval = 0, maxval = 1)
        # predict_reshaped = tf.reshape(predict, (-1, self.S, self.S, self.C + 5 * self.B))
        # get object proba and predict boxes, get class proba
        # proba_object = predict_reshaped[..., :980]
        # predict_boxes = predict_reshaped[..., 980:1078]
        # proba_classes = predict_reshaped[..., 1078:]
        predict_reshaped = tf.reshape(predict, (-1, self.S * self.S * (self.C + self.B * 5)))
        proba_classes = tf.reshape(predict_reshaped[:, :980], (-1, self.S, self.S, self.C))
        proba_object = tf.reshape(predict_reshaped[:, 980:1078], (-1, self.S, self.S, self.B))
        predict_boxes = tf.reshape(predict_reshaped[:, 1078:], (-1, self.S, self.S, self.B, 4))
        # calculate global x and y according respetive to grid cx and xy
        boxes_x = (predict_boxes[..., 0] + x_offset) / self.S
        boxes_y = (predict_boxes[..., 1] + y_offset) / self.S
        boxes_w = tf.square(predict_boxes[..., 2])
        boxes_h = tf.square(predict_boxes[..., 3])
        # get (x1, y1) and (x2, y2) of a rectangle box based on center point(x,y) and (w, h)
        boxes_x1 = tf.maximum(0, boxes_x - boxes_w / 2)
        boxes_x2 = tf.minimum(1, boxes_x + boxes_w / 2)
        boxes_y1 = tf.maximum(0, boxes_y - boxes_h / 2)
        boxes_y2 = tf.minimum(1, boxes_y + boxes_h / 2)
        # concat the cordinates into (..., (x1, y1, x2, y2))
        boxes_relative_global = tf.stack((boxes_x1, boxes_y1, boxes_x2, boxes_y2), axis=-1)
        # filtering the output with predict confidence
        proba_argmax_classes = tf.argmax(proba_classes, axis=-1)
        proba_max_classes = tf.reduce_max(proba_classes, axis=-1)
        proba_argmax_classes = tf.stack([proba_argmax_classes for i in range(self.B)], axis=-1)
        proba_max_classes = tf.stack([proba_max_classes for i in range(self.B)], axis=-1)
        proba = proba_max_classes * proba_object
        # filtering output using confidence threshold
        proba_filter_mask = proba >= self.confidence_threshold
        proba_filtered = tf.boolean_mask(proba, proba_filter_mask)
        boxes_filtered = tf.boolean_mask(boxes_relative_global, proba_filter_mask)
        class_filtered = tf.boolean_mask(proba_argmax_classes, proba_filter_mask)
        # self._nms_method == 'soft_nms'
        # assert isinstance(self._nms_method, str) == True

        if self.nms_method == 'nms':
            boxes_filtered, proba_filtered, class_filtered = self.nms(boxes_filtered, proba_filtered, class_filtered)
        elif self.nms_method == 'soft-nms':
            boxes_filtered, proba_filtered, class_filtered = self.soft_nms(
                boxes_filtered, proba_filtered, class_filtered)
        return boxes_filtered, proba_filtered, class_filtered

    def detect_from_image(self, img_file_path):
        # detect from a given image file path
        img = cv2.imread(img_file_path)
        img_cp = img.copy()
        img_shape = img.shape
        # get the output of the yolo net with input: image
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # mean = np.array([123, 117, 104], dtype=np.float32)
        # img = img - mean
        # img /= 255
        img_RGB = (img_RGB / 255) * 2.0 - 1.0

        img = np.expand_dims(img_RGB, axis=0).astype(np.float32)
        predict = self.forward(img)
        boxes, proba, classes = self._build_detector(predict, img_shape)
        self.visualize_predict(img_cp, boxes, proba, classes)

    def visualize_predict(self, image, boxes, proba, classes):
        # visualize boxes and labels proba and classes
        img_cp = image.copy()
        h, w, _ = img_cp.shape
        h, w = w, h
        num = boxes.shape[0]
        for i in range(num):
            box, p, c = boxes[i, :].numpy(), proba[i].numpy(), self.classes[classes[i].numpy()]
            x1, y1, x2, y2 = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)
            cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.rectangle(img_cp, (x1, y1 - 20), (x2, y1), (125, 125, 125), -1)
            cv2.putText(img_cp, '{} p={:.3f}'.format(c, p), (x1 + 1, y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite('./output/person_detected.jpg', img_cp)
        cv2.imshow('YOLO_small detection', img_cp)
        cv2.waitKey()


if __name__ == '__main__':
    yolo = YoloInf('./data/YOLO_small.ckpt')
    yolo.detect_from_image('./test/person.jpg')
