'''
使用onnx进行推理
'''
import onnxruntime
import torchvision.transforms as Trans
import numpy as np
import torchvision.transforms as Trans
import PIL.Image as Image
import matplotlib.pyplot as plt

def infer(onnx_path, img_path):
    im = Image.open(img_path)
    if im.split().__len__() == 1:
        im = im.convert('RGB')
    im = Trans.ToTensor()(im)
    im = im.unsqueeze(dim=0)
    im = np.asarray(im)
    sess = onnxruntime.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name:im})
    scores, masks = np.squeeze(outs[0]), np.squeeze(outs[1])
    
    keep = np.argmax(scores, axis=1)
    masks = [masks[label, :, :] for i, label in enumerate(keep) if scores[i, label] > 0.35]
    fig = plt.figure()
    num_masks = len(masks)
    for i, mask in enumerate(masks, 1):
        fig.add_subplot(1, num_masks, i)
        plt.imshow(mask)
    plt.show()
    plt.ion()

if __name__ == '__main__':
    onnx_path = './output/sparse_inst_r50_giam/model_0009999.onnx'
    img_path = './datasets/coco-608-608-val/images/7.jpg'
    infer(onnx_path, img_path)