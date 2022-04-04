package com.mat.multiplication.strassen;

public class StrassenMatrixMultiplicationAlgorithm {
	public float[][] strassen(float[][] a, float[][] b) {

		StrassenSplitAddSubJoin SASJ = new StrassenSplitAddSubJoin();
		int n = a.length; // Calulating the size of the matrix
		float[][] R = new float[n][n]; //Resultant Matrix Initialisation
		if (n == 1)
			R[0][0] = a[0][0] * b[0][0];
		else if (n <= 128) // Threshold Value is set to 128 same as that of OpenMP
		{
			float[][] a11 = new float[n / 2][n / 2];
			float[][] a12 = new float[n / 2][n / 2];
			float[][] a21 = new float[n / 2][n / 2];
			float[][] a22 = new float[n / 2][n / 2];
			float[][] b11 = new float[n / 2][n / 2];
			float[][] b12 = new float[n / 2][n / 2];
			float[][] b21 = new float[n / 2][n / 2];
			float[][] b22 = new float[n / 2][n / 2];

			// Matrix DIvision into 4 quads.
			SASJ.split(a, a11, 0, 0);
			SASJ.split(a, a12, 0, n / 2);
			SASJ.split(a, a21, n / 2, 0);
			SASJ.split(a, a22, n / 2, n / 2);

			// Matrix DIvision into 4 quads.
			SASJ.split(b, b11, 0, 0);
			SASJ.split(b, b12, 0, n / 2);
			SASJ.split(b, b21, n / 2, 0);
			SASJ.split(b, b22, n / 2, n / 2);

			float[][] M1 = strassen(SASJ.add(a11, a22), SASJ.add(b11, b22));
			float[][] M2 = strassen(SASJ.add(a21, a22), b11);
			float[][] M3 = strassen(a11, SASJ.sub(b12, b22));
			float[][] M4 = strassen(a22, SASJ.sub(b21, b11));
			float[][] M5 = strassen(SASJ.add(a11, a12), b22);
			float[][] M6 = strassen(SASJ.sub(a21, a11), SASJ.add(b11, b12));
			float[][] M7 = strassen(SASJ.sub(a12, a22), SASJ.add(b21, b22));

			float[][] C11 = SASJ.add(SASJ.sub(SASJ.add(M1, M4), M5), M7);
			float[][] C12 = SASJ.add(M3, M5);
			float[][] C21 = SASJ.add(M2, M4);
			float[][] C22 = SASJ.add(SASJ.sub(SASJ.add(M1, M3), M2), M6);

			//Joining 4 quads of the matrices intp the Resultant matrix R.
			SASJ.join(C11, R, 0, 0);
			SASJ.join(C12, R, 0, n / 2);
			SASJ.join(C21, R, n / 2, 0);
			SASJ.join(C22, R, n / 2, n / 2);

		}

		else { 
			
			//If matrix is greater than the threshold size that it 128 then the dense matrix multiplication starts using the threads.
			float[][] a11 = new float[n / 2][n / 2];
			float[][] a12 = new float[n / 2][n / 2];
			float[][] a21 = new float[n / 2][n / 2];
			float[][] a22 = new float[n / 2][n / 2];
			float[][] b11 = new float[n / 2][n / 2];
			float[][] b12 = new float[n / 2][n / 2];
			float[][] b21 = new float[n / 2][n / 2];
			float[][] b22 = new float[n / 2][n / 2];

			class Mul implements Runnable {
				private Thread t;
				private String threadName;
				StrassenSplitAddSubJoin SASJ = new StrassenSplitAddSubJoin();

				Mul(String name) {
					threadName = name;
				}

				public void run() {
					
					//Splitting and Dividing the first matrix into 4 halves
					if (threadName.equals("thread1")) {
						SASJ.split(a, a11, 0, 0);
					} else if (threadName.equals("thread2")) {
						SASJ.split(a, a12, 0, n / 2);
					}

					else if (threadName.equals("thread3")) {
						SASJ.split(a, a21, n / 2, 0);
					}

					else if (threadName.equals("thread4")) {
						SASJ.split(a, a22, n / 2, n / 2);
					}
                    
					//Dividing the second matrix into 4 halves
					else if (threadName.equals("thread5")) {
						SASJ.split(b, b11, 0, 0);
					}

					else if (threadName.equals("thread6")) {
						SASJ.split(b, b12, 0, n / 2);
					}

					else if (threadName.equals("thread7")) {
						SASJ.split(b, b21, n / 2, 0);
					}

					else if (threadName.equals("thread8")) {
						SASJ.split(b, b22, n / 2, n / 2);
					}
				}

				public void start() {
					if (t == null) {
						t = new Thread(this, "Thread1");
						t.start();
					}
				}
			}
			
			//Assignment of threads to every step at Splitting in the algorithm
			Mul m1 = new Mul("thread1");
			m1.start();
			Mul m2 = new Mul("thread2");
			m2.start();
			Mul m3 = new Mul("thread3");
			m3.start();
			Mul m4 = new Mul("thread4");
			m4.start();
			Mul m5 = new Mul("thread5");
			m5.start();
			Mul m6 = new Mul("thread6");
			m6.start();
			Mul m7 = new Mul("thread7");
			m7.start();
			Mul m8 = new Mul("thread8");
			m8.start();
			
			//Arranging and looping all the threads. 
			for (Thread t : new Thread[] { m1.t, m2.t, m3.t, m4.t, m5.t, m6.t, m7.t, m8.t }) {
				try {
					t.join(); //Join helps to execute thread relative to one another.
				} catch (Exception e) {
					System.out.println(e);
				}
			}

			float[][] M1 = strassen(SASJ.add(a11, a22), SASJ.add(b11, b22));
			float[][] M2 = strassen(SASJ.add(a21, a22), b11);
			float[][] M3 = strassen(a11, SASJ.sub(b12, b22));
			float[][] M4 = strassen(a22, SASJ.sub(b21, b11));
			float[][] M5 = strassen(SASJ.add(a11, a12), b22);
			float[][] M6 = strassen(SASJ.sub(a21, a11), SASJ.add(b11, b12));
			float[][] M7 = strassen(SASJ.sub(a12, a22), SASJ.add(b21, b22));

			float[][] C11 = SASJ.add(SASJ.sub(SASJ.add(M1, M4), M5), M7);
			float[][] C12 = SASJ.add(M3, M5);
			float[][] C21 = SASJ.add(M2, M4);
			float[][] C22 = SASJ.add(SASJ.sub(SASJ.add(M1, M3), M2), M6);

			class Mul2 implements Runnable {
				private Thread t;
				private String threadName;
				StrassenSplitAddSubJoin SASJ = new StrassenSplitAddSubJoin();

				Mul2(String name) {
					threadName = name;
				}
               
				//Assigning threads to every step in Join
				public void run() {
					if (threadName.equals("thread1")) {
						SASJ.join(C11, R, 0, 0);
					} else if (threadName.equals("thread2")) {
						SASJ.join(C12, R, 0, n / 2);
					}

					else if (threadName.equals("thread3")) {
						SASJ.join(C21, R, n / 2, 0);

					}

					else if (threadName.equals("thread4")) {
						SASJ.join(C22, R, n / 2, n / 2);
					}

				}

				//Thread execution starts here.
				public void start() {
					if (t == null) {
						t = new Thread(this, "Thread1");
						t.start();
					}
				}
			}

			Mul2 m11 = new Mul2("thread1");
			m11.start();
			Mul2 m22 = new Mul2("thread2");
			m22.start();
			Mul2 m33 = new Mul2("thread3");
			m33.start();
			Mul2 m44 = new Mul2("thread4");
			m44.start();
			
			//Arranging and looping all the threads.
			for (Thread t : new Thread[] { m11.t, m22.t, m33.t, m44.t }) {
				try {
					t.join();
				} catch (Exception e) {
					System.out.println(e);
				}
			}
		}
		return R;
	}

}
