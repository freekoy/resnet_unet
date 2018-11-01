from data import *
from resnet50_unet import ResNet50_Unet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
size = 256
input_shape = (size,size,1)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = None,target_size=(size,size))
# Don't fill in the weights, it won't work
model = ResNet50_Unet(weights=None, include_top=False, input_shape=input_shape)
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test",target_size=(size,size))
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)