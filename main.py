import sys
import PyQt5
from PyQt5 import QtWidgets 
from PyQt5 import QtGui 
from PyQt5 import QtCore 

from mainwindow import Ui_MainWindow


class Main(QtWidgets.QMainWindow, Ui_MainWindow):

	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.loadVidButton.clicked.connect(self.open_file)

	def open_file(self):
		options = QtWidgets.QFileDialog.Options()
		options |= QtWidgets.QFileDialog.DontUseNativeDialog
		fname, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", '.', "(*.mp4 *.mpg *.mkv *.avi)")
		if fname:
			self.ui.file_path = fname
			self.ui.setFilePathLabel(fname)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
