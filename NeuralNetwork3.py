import sys
import numpy as np
import time
np.random.seed(49) #seeding the random to avoid deviation in result

'''Array generation from input csv file'''
def data_loading(f1,f2,f3):
    #converting csv to numpy array
    train_image = np.genfromtxt(f1, delimiter= ",")
    train_label = np.genfromtxt(f2, delimiter= ",")
    test_image = np.genfromtxt(f3, delimiter= ",")

    #normalizing the array so that the value is between 0 and 1
    train_image = train_image.astype(float)/255.
    test_image = test_image.astype(float)/255.

    #converting to int
    train_label = train_label.astype(int)

    #making it to nx1 array
    train_image = train_image.reshape(train_image.shape[0],-1)
    test_image = test_image.reshape(test_image.shape[0],-1)

    return train_image,train_label,test_image

'''Batch training'''
def splitbatches(train_image,train_label,batch):
    for start in range(0, len(train_image) , batch):
        index = slice(start, start + batch)
        yield train_image[index], train_label[index]

'''Class for Hidden Layer'''
class Hidden():
    def __init__(self,input,output):
        self.weight = np.random.randn(input,output) * np.sqrt(2/(input+output)) #xavier initialization
        self.bias = np.zeros(output)

    def forward(self,train_image):
        return np.dot(train_image,self.weight) + self.bias

    def backward(self,train_image,train_label):
        learn_rate = 0.15
        gradient_input = np.dot(train_label,self.weight.T)
        gradient_weight = np.dot(train_image.T, train_label)
        gradient_bias = train_label.mean(axis=0)*train_image.shape[0]
        self.weight = self.weight - learn_rate * gradient_weight #updating the weight
        self.bias = self.bias - learn_rate * gradient_bias #updating the bias
        return gradient_input

'''Class for Activation Layer(Function)'''
class LeakyReLU():
    def forward(self,input):
        alpha = 0.01
        return np.maximum(alpha*input,input)

    def backward(self,input,output):
        true_input = input > 0
        return (true_input*output)

'''Class for Network'''
class Network():
    def __init__(self):
        self.network = list()
        self.network.append(Hidden(784,200)) #hidden 1
        self.network.append(LeakyReLU()) #activation 1
        self.network.append(Hidden(200,300)) #hidden 2
        self.network.append(LeakyReLU()) #activation 2
        self.network.append(Hidden(300,10)) #output

    '''Softmax function'''
    def softmax(self,input):
        exponent = np.exp(input)
        return exponent/np.sum(exponent, axis = 1, keepdims = True)

    '''Loss(Cost) function'''
    def cross_entropy_loss(self,obtained_output,expected_output):
        c = expected_output.shape[0]
        soft = self.softmax(obtained_output) #probability of the obtained input
        log_cal = -np.log(soft[range(c),expected_output])
        loss = np.sum(log_cal)/c #loss equation
        return loss

    def gradient_cross_entropy(self,obtained_input,expected_output):
        c = obtained_input.shape[0]
        logits = np.zeros((obtained_input.shape[0],obtained_input.shape[1]))
        logits[range(c),expected_output] = 1 #assigning value=1 for given label index
        soft = self.softmax(obtained_input) #probability of the obtained input
        return (- logits + soft)/c

    '''Forward calculation'''
    def forward(self,train_image):
        activation = []

        for each in self.network:
            activation.append(each.forward(train_image))
            train_image = activation[-1]
        return activation

    '''Training the dataset'''
    def train(self, train_image, train_label):
        list_of_activation = self.forward(train_image)
        obtained_output = list_of_activation[-1]
        layer_with_activation = [train_image]+list_of_activation #Adding the given data to the activation found  
        loss_grad = self.gradient_cross_entropy(obtained_output,train_label) 

        for index in range(len(self.network))[::-1]:
            current = self.network[index]
            loss_grad = current.backward(layer_with_activation[index],loss_grad) #backward calculation

        loss = self.cross_entropy_loss(obtained_output,train_label) #loss calculation ~ not used anywhere

    '''Predicting the given image'''
    def predict(self,test_image):
        output = []
        input = test_image
        for i in self.network:
            output.append(i.forward(input))
            input = output[-1]
        res = self.softmax(output[-1])
        return res.argmax(axis=-1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train_image,train_label,test_image = data_loading('train_image.csv','train_label.csv','test_image.csv')
    else:
        f1,f2,f3 = sys.argv[1:4]
        train_image,train_label,test_image = data_loading(f1,f2,f3)
    net_object = Network()
    for i in range(22):
        for x,y in splitbatches(train_image,train_label,batch = 50):
            net_object.train(x,y)
    result = net_object.predict(test_image) #predicted result obtained as array
    f = open("test_predictions.csv","w") #file writer
    f.write("\n".join([str(x) for x in result]))
    f.close()