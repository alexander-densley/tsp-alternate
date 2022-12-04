from TSPSolver import TSPSolver
from Proj5GUI import *

NUM_TESTS = 5

def testFancyGreedyWithCurrentConditions(w:Proj5GUI):
	w.generateClicked()
	print(w.curSeed.text(), w.size.text(), sep=", ", end=", ")
	w.algDropDown.setCurrentIndex(3)
	w.solveClicked()
	print(w.tourCost.text(), float(w.solvedIn.text().split()[0]), sep=", ", end=", ")
	w.algDropDown.setCurrentIndex(1)
	w.solveClicked()
	print(w.tourCost.text(), float(w.solvedIn.text().split()[0]), sep=", ")

if __name__ == '__main__':
	# This line allows CNTL-C in the terminal to kill the program
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	
	app = QApplication(sys.argv)
	w:Proj5GUI = Proj5GUI()
	w.algDropDown.setCurrentIndex(3)	#set to fancy
	w.diffDropDown.setCurrentIndex(2)	#set to Hard (not deterministic)

	original_stdout = sys.stdout # Save a reference to the original standard output
	with open('data.csv', 'w') as f:
		sys.stdout = f
		print("Current Seed, Problem Size, Fancy Cost, Fancy Time (sec), Greedy Cost, Greedy Time (sec)")
		for i in range(10, 201, 10):
			w.size.setText(str(i))
			for i in range(NUM_TESTS):
				# w.size.setText("20") # set size
				w.randSeedClicked()
				testFancyGreedyWithCurrentConditions(w)
	sys.exit(app.exec())

