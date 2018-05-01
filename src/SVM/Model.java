package SVM;

public class Model<W,B,K> {
	private W w;
	private B b;
	private K k;
	public Model(){
		
	}
	public Model(W w,B b,K k){
		this.w=w;
		this.b=b;
		this.k=k;
	}
	public K getKernel(){
		return k;
	}
	public W getW(){
		return w;
	}
	public B getB(){
		return b;
	}
}