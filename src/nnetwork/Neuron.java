package nnetwork;
import java.io.Serializable;

public class Neuron implements Serializable{
	private static final long serialVersionUID = -9136876609916059653L;
	
	double value = 0;
	double error = -1;
	
	int count = 0, size = 0;
	Neuron[] previousNeurons;
	double[] previousDoubles;
	
	public Neuron(int size) {
		this.size = size;
		previousNeurons = new Neuron[size];
		previousDoubles = new double[size];
	}
	
	public void connect(Neuron n) {
		previousNeurons[count] = n;
		previousDoubles[count++] = Math.random() * 2 - 1;
	}

	public void compute() {
		double total = 0;
		for (int i = 0; i < size; i++) 
			total = total + previousNeurons[i].value * previousDoubles[i];

		value = 1 / (1 + Math.pow(Math.E, -total) );
	}

}
