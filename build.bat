cd C:\Users\MPardo\Documents\PyDocs\FilmProcesser
call conda activate nomkl
pyinstaller FilmProcesser.py --noconfirm ^
	--paths="C:\Users\MPardo\miniconda3\envs\nomkl\Lib\site-packages\cv2" ^
	--icon=icon.ico ^
	--onefile
pyinstaller setup.py --noconfirm ^
	--icon=icon.ico ^
	--onefile
call conda deactivate
exit