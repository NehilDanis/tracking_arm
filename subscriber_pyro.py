import torch
import Pyro5
import Pyro5.api

import numpy as np
import cv2



import msgpack
import msgpack_numpy as m
m.patch()
Pyro5.config.SERIALIZER = "msgpack"
Pyro5.config.SERVERTYPE = "multiplex"

@Pyro5.api.expose
@Pyro5.server.behavior(instance_mode="single")
class SegmentArm(object):
    def __init__(self):
        # create the model using the given model_path
        # Load the trained model
        self.model = torch.load('/home/nehil/arm_w_tarso_data_folder/weights_vgg16.pt')
        # Set the model to evaluate mode
        self.model.eval()

    def segment_arm(self, img):

        img = img[:, :, :3]
        print("start arm segmentation")
        # the image coming from the camera is BGR --> make it RGB
        # change the size to (1, 3, 512, 512)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("original_img", img)
        cv2.waitKey(1)
        img = cv2.resize(img, (512, 512)).transpose(2, 0, 1).reshape(1, 3, 512, 512)
        #print(img.shape)


        with torch.no_grad():
            a = self.model(torch.from_numpy(img).type(torch.cuda.FloatTensor) / 255)

        mask = np.array(a.cpu().detach().numpy()[0][0]>0.2) * 1


        img = img.reshape(512, 512, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #cv2.bitwise_and(img, img, mask = mask)
        img[mask==1] = 255
        img[mask==0] = 0


        cv2.imshow("mask", img)
        cv2.waitKey(1)

        return img


if __name__ == '__main__':
    #arm_segment = SegmentArm()
    # for example purposes we will access the daemon and name server ourselves and not use serveSimple

    daemon = Pyro5.server.Daemon()  # make a Pyro daemon
    ns = Pyro5.api.locate_ns()  # find the name server
    uri = daemon.register(SegmentArm)  # register the greeting maker as a Pyro object
    ns.register("segmentation.arm_segment", uri)  # register the object with a name in the name server

    print("Ready.")
    daemon.requestLoop()
