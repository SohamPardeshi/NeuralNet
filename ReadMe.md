# NeuralNetwork
This is a simple feedforward neural network (with backpropagation) built from scratch using Java. It can be used to approximate many tasks, such as classifying MNIST images with ~93% accuracy or identifying generic machine data.


### Setup
Begin by specifying the size of each input, hidden, and output layer in the Network.
```java
int   inputSize   = 28 * 28;		 // Declare an input size (e.g. for a 28x28 pixel MNIST image)
int[] hiddenSizes = {40, 20, 15};	 // Specify the sizes of each hidden layer (3 layers)
int   outputSize  = 10;				 // Choose the number of output classifications

Network net = new Network(inputSize, hiddenSizes, outputSize);
```

Begin training your data...
```java
for(int i = 0; i < trainingSet.size(); i++){
	double[] input = trainingSet.getTest(i);
	int label = trainingSet.getLabel(i);
	
	double[] expected = NNetworkUtils.oneHotArray(outputSize, label);
	double learningRate = 0.1;
	
	net.compute(input);
	net.backpropagate(expected, learningRate);
}
```


... and lastly test your training.
```java
int correct = 0;
for(int i = 0; i < testSet.size(); i++){
	double[] input = testSet.getTest(i);
	int label = trainingSet.getLabel(i);
	
	double[] output = net.compute(input);
	
	int guess = NNetworkUtils.softmax(output);
	if(guess == label) correct++;
	
}
System.out.println("Classified " + correct + " out of " + testSet.size());
```