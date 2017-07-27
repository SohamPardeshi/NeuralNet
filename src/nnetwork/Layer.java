package nnetwork;
import java.io.Serializable;

public class Layer implements Serializable {
	private static final long serialVersionUID = 2542188198852026067L;
	
	Neuron[] neurons;
	Layer prev, next;

	public Layer(int size, int prevSize) {
		neurons = new Neuron[size];
		for(int i = 0; i < size; i++)
			neurons[i] = new Neuron(prevSize);
	}

	public void connect(Layer prev) {
		this.prev = prev;
		prev.next = this;
		for(Neuron n : neurons)
			for(Neuron pn : prev.neurons)
				n.connect(pn);
	}

	public void compute() {
		for(Neuron n : neurons)
			n.compute();
	}
	
	public void set(double[] values) {
		assert values.length == neurons.length;
		for(int i = 0; i < values.length; i++)
			neurons[i].value = values[i];
	}

	public double[] get() {
		double[] arr = new double[neurons.length];
		for(int i = 0; i < arr.length; i++)
			arr[i] = neurons[i].value;
		return arr;
	}
}
