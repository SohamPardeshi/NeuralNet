package nnetwork;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;

public class Network implements Serializable{
	private static final long serialVersionUID = 728708967403252047L;

	Layer input, output;
	Layer[] hiddens;

	public Network(int inputSize, int[] hiddenSize, int outputSize) {
		assert hiddenSize.length > 0;
		input = new Layer(inputSize, 0);
		hiddens = new Layer[hiddenSize.length];
		hiddens[0] = new Layer(hiddenSize[0], inputSize);

		// Go through each one
		for(int i = 1; i < hiddens.length; i++)
			hiddens[i] = new Layer(hiddenSize[i], hiddenSize[i - 1]);

		output = new Layer(outputSize, hiddenSize[hiddenSize.length - 1]);

		hiddens[0].connect(input);
		for(int i = 1; i < hiddens.length; i++)
			hiddens[i].connect(hiddens[i - 1]);
		output.connect(hiddens[hiddens.length - 1]);
	}

	public double[] compute(double[] in) {
		input.set(in);
		for(Layer hidden : hiddens)
			hidden.compute();
		output.compute();
		return output.get();
	}

	public void backpropagate(double[] expected, double rate) {

		/** Output Layer */
		for(int i = 0; i < output.neurons.length; i++) {
			Neuron outputNeuron = output.neurons[i];
			outputNeuron.error = expected[i] -  outputNeuron.value;
			gradientDescent(outputNeuron, rate);
		}

		/** HIDDEN LAYERS */

		// Get a reversed copy of the hidden layers
		Layer[] reverseHiddens = Arrays.copyOf(hiddens, hiddens.length);
		Collections.reverse(Arrays.asList(reverseHiddens));
				
		for(Layer hLayer : reverseHiddens) {
			// Run the backpropagation of each hidden Layer
			for(Neuron hiddenNeuron : hLayer.neurons) {
				// Sum the partial derivative error rates of all previous errors 
				hiddenNeuron.error = 0;
				for(int j = 0; j < hLayer.next.neurons.length; j++) {
					Neuron outputNeuron = hLayer.next.neurons[j];
					double outputValue = outputNeuron.value;
					double weight =  0;

					// Equivalent of HashMap<Neuron, Double>::get(Key) 
					for(int tmp = 0; tmp < outputNeuron.size; tmp++)
						if(outputNeuron.previousNeurons[tmp] == hiddenNeuron) 
							weight = outputNeuron.previousDoubles[tmp];

					hiddenNeuron.error += outputValue * (1 - outputValue) * outputNeuron.error * weight;
				}

				gradientDescent(hiddenNeuron, rate);
			}	
		}
	}

	private void gradientDescent(Neuron outputNeuron, double rate) {
		// Calculate the gradient descent slope of each hidden input
		for(int tmp = 0; tmp < outputNeuron.size; tmp++) {
			Neuron neuron = outputNeuron.previousNeurons[tmp];
			double value = neuron.value;
			double weight = outputNeuron.previousDoubles[tmp];

			double slope = -outputNeuron.value * (1 - outputNeuron.value) * value * outputNeuron.error;
			double updated = weight - rate * slope;
			outputNeuron.previousDoubles[tmp] = updated;
		}
	} 
}
