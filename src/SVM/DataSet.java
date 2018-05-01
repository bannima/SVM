package SVM;

public class DataSet<K,A> {
	private K k;
	private A a;
	//private P p;
	public DataSet(){
		
	}
	public DataSet(K k,A a){
		this.k=k;
		this.a=a;
	}
//	public DataSet(K k,A a,P p){
//		this.k=k;
//		this.a=a;
//		this.p=p;
//	}
//	public P getKerProduct(){
//		return p;
//	}
	public K getData(){
		return k;
	}
	public A getLabel(){
		return a;
	}
}
