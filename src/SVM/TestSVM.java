package SVM;
import java.io.*;
import cern.colt.*;
import cern.colt.matrix.*;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.*;


public class TestSVM {
	public static void main(String[] args) throws Exception{
		try {
			//load dataset
			SVM svm = new SVM();
			DataSet<DoubleMatrix2D,DoubleMatrix1D>dataset = svm.loadDataSet("testSet.txt");
			//train svm classifier
			int maxiter = 40;
			double C = 0.6;
			double toler = 0.001;
			String kernelType = "liner";
			Model<DoubleMatrix1D, Double,String> model = svm.PlattSMO(dataset, C, toler, maxiter,kernelType);
			//show parameters
			System.out.println(model.getW());
			System.out.println(model.getB());
			//predict result
			double accuracy = svm.predict(model, dataset);
			System.out.println("accuracy: "+accuracy*100+" %");
			//visualize nodes
			View view = new View();
			
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
}
