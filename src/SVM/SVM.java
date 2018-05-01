package SVM;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import cern.colt.*;
import cern.colt.matrix.*;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.*;

public class SVM {
	private Model model;

	public SVM() {

	}

	public SVM(Model model) {
		this.model = model;
	}

	//predict result
	public double predict(Model model,DataSet dataset) throws Exception{
		//model
		DoubleMatrix2D data = (DoubleMatrix2D) dataset.getData();
		DoubleMatrix1D label = (DoubleMatrix1D) dataset.getLabel();
		
		//init k
		int m = label.size();
		DoubleMatrix2D kerProduct = new DenseDoubleMatrix2D(m,m);
		for(int i=0;i<m;i++){
			for(int j=0;j<m;j++){
				kerProduct.set(i,j, kernelProduct(data.viewRow(i),data.viewRow(j),(String)model.getKernel()));
			}
		}
		//classifier
		DoubleMatrix1D alphas = (DoubleMatrix1D)model.getW();
		double b = (Double)model.getB();
		int count = 0;
		for(int i=0;i<m;i++){
			//predict data[i]
			double pred = 0;
			double result = 0;
			for(int j=0;j<m;j++){
				pred+=alphas.get(j)*label.get(j)*kerProduct.get(i, j);
			}
			pred=pred+b;
			if(pred>0){
				result = 1;
			}
			else{
				result = -1;
			}
			//calc correct rate
			if(result==label.get(i)){
				count +=1;
			}
			//System.out.println("count: "+count+" pred: "+pred+" label:"+label.get(i)+" result: "+result);
		}
		//output accuracy
		System.out.println("correct: "+count+" total cases: "+m);
		double accuracy = (count/(double)m);
		return accuracy;
	}
	

	// train svm model using SMO algorithm
	public Model<DoubleMatrix1D, Double,String> PlattSMO(DataSet data, double C, double toler, int maxiter,String kernelType) throws Exception {
		Model<DoubleMatrix1D, Double,String> model;
		DoubleMatrix2D dataset = (DoubleMatrix2D) data.getData();
		DoubleMatrix1D label = (DoubleMatrix1D) data.getLabel();
		
		// train model
		int iter = 0;
		int m = dataset.rows();
		DoubleMatrix1D alphas = new DenseDoubleMatrix1D(m);
		double b = (double) 0.0;
		DoubleMatrix1D E = new DenseDoubleMatrix1D(m);
		DoubleMatrix2D K = new DenseDoubleMatrix2D(m, m);
		// init Kij with kernel type
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				K.set(i, j, kernelProduct(dataset.viewRow(i), dataset.viewRow(j), kernelType));
			}
		}
		//init E with k
		for(int i=0;i<m;i++){
			double predi,Ei;
			predi = calcPredk(alphas, label, K, i, b);
			Ei = predi - label.get(i);
			E.set(i, Ei);
		}
		
		// show K
		//System.out.println(K);

		// calc SMO
		while (iter < maxiter) {
			System.out.println("iter: " + iter);
			// choose i
			Boolean changeFlag=false;
			for (int i = 0; i < m; i++) {
				double predi,Ei,newEi;
				changeFlag = false;
				predi = calcPredk(alphas, label, K, i, b);
				Ei = predi - label.get(i);
				E.set(i, Ei);
				
				// choose the break kkt condition i and train svm
				if ((alphas.get(i) < C && label.get(i) * E.get(i) < -1 * toler)
						|| (alphas.get(i) > 0 && label.get(i) * E.get(i) > toler)) {
					System.out.println("kkt condition satisfied " + i);
					// choose random j according to i
					int j = randJ(i, m, E,alphas);
					double predj,Ej;
					double alphaJNew,alphaJOld,alphaINew,alphaIOld;
					double L,H,eta;
					// calc Ei and Ej and update Ej,Ei
					System.out.println(j);
					predj = calcPredk(alphas, label, K, j, b);
					Ej = predj - label.get(j);
					E.set(j, Ej);

					alphaIOld = alphas.get(i);
					alphaJOld = alphas.get(j);

					eta = -2 * K.get(i, j) + K.get(i, i) + K.get(j, j);
					if (eta <= 0) {
						System.out.println("eta <= 0 Error");
						continue;
					}
					alphaJNew = alphaJOld + (label.get(j) * (E.get(i) - E.get(j)) /eta);
					if (label.get(i) == label.get(j)) {
						L = Math.max(0, alphaJOld + alphaIOld - C);
						H = Math.min(C, alphaJOld + alphaIOld);
					} else {
						L = Math.max(0, alphaJOld - alphaIOld);
						H = Math.min(C, C + alphaJOld - alphaIOld);
					}
					if (L == H) {
						System.out.println("ERROR: L==H");
						continue;
					}
					alphaJNew = cutAlpha(alphaJNew, L, H);

					if (Math.abs(alphaJOld - alphaJNew) < 0.0001) {
						System.out.println("Warning: j not move enough/ j: "+j+" i: "+i+" iter: "+iter);
						//Thread.sleep(1000);
						continue;
					}

					alphaINew = alphaIOld + label.get(i) * label.get(j) * (alphaJOld - alphaJNew);
					// save ai and aj
					alphas.set(i, alphaINew);
					alphas.set(j, alphaJNew);
					// calc new b and Ek
					double b1new = -1 * E.get(i) - label.get(i) * K.get(i, i) * (alphaINew - alphaIOld)
							- label.get(j) * K.get(j, i) * (alphaJNew - alphaJOld) + b;
					double b2new = -1 * E.get(j) - label.get(i) * (alphaINew - alphaIOld) * K.get(i, j)
							- label.get(j) * (alphaJNew - alphaJOld) * K.get(j, j) + b;
					//update new b
					if (0 < alphas.get(i) && (alphas.get(i) < C)) {
						b = b1new;
					} else if (0 < alphas.get(j) && (alphas.get(j) < C)) {
						b = b2new;
					} else {
						b = (b1new + b2new) / 2.0;
					}
					// update new Ei
					predi = calcPredk(alphas, label, K, i, b);
					newEi = predi - label.get(i);
					E.set(i, newEi);
					// calc Ei and Ej and update Ej,Ei
					predj = calcPredk(alphas, label, K, j, b);
					Ej = predj - label.get(j);
					E.set(j, Ej);
					//alpha pair changed
					changeFlag = true;
				} // end if kkt condition
				
			} // end for
			//judge to add iter number when all cases are null
			if (changeFlag==false) {
				iter += 1;
			}else{
				iter=0;
			}
			System.out.println("iternumber " + iter);

		} // end while
		model = new Model<DoubleMatrix1D, Double,String>(alphas, b,kernelType);
		//save model to this svm
		this.model = model;
		return model;
	}

	// cut alpha
	public double cutAlpha(double alpha, double low, double high) {
		if (alpha > high) {
			return high;
		}
		if (alpha < high && alpha > low) {
			return alpha;
		}
		if (alpha < low) {
			return low;
		}
		return -1;
	}

	// calc Ek
	public double calcPredk(DoubleMatrix1D alphas, DoubleMatrix1D label, DoubleMatrix2D K, int k, double b) {
		double res = 0;
		int m = alphas.size();
		for (int i = 0; i < m; i++) {
			res += alphas.get(i) * label.get(i) * K.get(k, i);
		}
		return res + b;
	}

	// kernel methods
	public double kernelProduct(DoubleMatrix1D a, DoubleMatrix1D b, String kind) throws Exception {
		double res = 0;
		switch (kind) {
		case "liner":
			res = Algebra.DEFAULT.mult(a, b);
			break;
		case "rbf":
			res = RBFKernel(a, b, 0.1);
			break;
		default:
			res = Algebra.DEFAULT.mult(a, b);
		}
		return res;
	}

	public double RBFKernel(DoubleMatrix1D a, DoubleMatrix1D b, double eta) throws Exception {
		double res = 0;
		if (a.size() != b.size()) {
			System.out.println("Error: multiply doublematrix1d dosen't match");
			throw new Exception("Error: not match doublematrix1d ");
		}
		int s = a.size();
		for (int i = 0; i < s; i++) {
			res += Math.pow(a.get(i) - b.get(i), 2);
		}
		res = Math.exp(-1 * res / Math.pow(eta, 2));
		return res;
	}

	// multiply
	public DoubleMatrix1D multiply(DoubleMatrix1D a, DoubleMatrix1D b) throws Exception {
		if (a.size() != b.size()) {
			System.out.println("Error: multiply doublematrix1d dosen't match");
			throw new Exception("Error: not match doublematrix1d ");
		}
		int m = a.size();
		DoubleMatrix1D res = new DenseDoubleMatrix1D(m);
		for (int i = 0; i < m; i++) {
			res.set(i, a.get(i) * b.get(i));
		}
		return res;
	}

	// choose random j according to max(|Ei-Ej|)
	public int randJ(int i, int m,DoubleMatrix1D E,DoubleMatrix1D alphas) {
//		int maxj=i;
//		while(maxj==i){
//			maxj=(int)(Math.random()*m);
//		}
//		return maxj;
		
		int maxj=-1;
		double maxDelta = -1;
		//exists nonzero alpha
		for(int j=0;j<m;j++){
			if(alphas.get(i)!=0){
				double delta = Math.abs(E.get(i)-E.get(j));
				if(delta>maxDelta){
					maxj=j;
					maxDelta = delta;
				}
			}
		}
		//all alphas is 0 and return random j
		if(maxj==-1){
			maxj=i;
			while(maxj==i){
				maxj=(int)(Math.random()*m);
			}
		}
		return maxj;
//		int maxj=-1;
//		if(maxj==-1){
//			double maxDelta = -1;
//			double delta;
//			for(int j=0;j<m;j++){
//				delta = Math.abs(E.get(j)-E.get(i));
//				if(delta>maxDelta){
//					maxj = j;
//					maxDelta = delta;
//				}
//			}
//			return maxj;
//		}
//		else{
//			maxj = i;
//			while (maxj == i) {
//				maxj = (int) (Math.random() * m);
//			}
//		}
//		return maxj;
		
	}

	// load data and labels
	public DataSet<DoubleMatrix2D, DoubleMatrix1D> loadDataSet(String filename) throws IOException {
		// DataSet<float[][],float[]> d = new DataSet<float[][],float[]>();
		// float[][] data = new float[100][2];
		DoubleMatrix2D data = new DenseDoubleMatrix2D(100, 2);
		DoubleMatrix1D label = new DenseDoubleMatrix1D(100);
		// float[] label = new float[100];
		// read file
		String path = "./data/";
		String pathfile = path + filename;
		InputStreamReader reader = new InputStreamReader(new FileInputStream(pathfile));
		BufferedReader br = new BufferedReader(reader);
		String line = "";
		line = br.readLine();
		int count = 0;
		while (line != null) {
			// System.out.println(line);
			// System.out.println(count);
			String[] array = line.split("\t");
			// System.out.println(array[0]);
			data.set(count, 0, Double.parseDouble(array[0]));
			data.set(count, 1, Double.parseDouble(array[1]));
			label.set(count, Double.parseDouble(array[2]));
			// data[count][0] = Float.parseFloat(array[0]);
			// data[count][1] = Float.parseFloat(array[1]);
			// label[count] = Float.parseFloat(array[2]);
			line = br.readLine();
			count += 1;
		}
		// System.out.println(count+" ");

		return new DataSet<DoubleMatrix2D, DoubleMatrix1D>(data, label);
	}
}
