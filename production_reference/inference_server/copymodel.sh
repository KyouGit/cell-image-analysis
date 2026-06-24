if [ -z $1 ] || [ $1 == "add" ] && [ -z $2 ] || [ $1 == "del" ] && [ -z $2 ]; then
	echo ""
	echo "copymodel - 모델 저장소에서 지정된 모델을 git 저장소로 복사한 뒤 필요 없는 파일을 정리 합니다"
	echo ""
	echo "Usage   : copymodel.sh [Command] [ModelName]"
	echo "	  [Command] := add | del | push"
	echo "	  [ModelName] := Model Name"
	echo "Example : copymodel.sh add 230922_U2L_001"
	echo "	  copymodel.sh del 230922_U2L_001"
	echo "	  copymodel.sh push"
	echo ""
else
	COMMAND=$1
	MODELNAME=$2

	if [ $COMMAND == "add" ]; then		
		rm yolo/model/*.jpg -f
		rm yolo/model/CITATION.cff -f
		rm yolo/model/LICENSE -f
		rm yolo/model/README.* -f
		rm yolo/model/tutorial.ipynb -f
		rm yolo/model/CONTRIBUTING.md -f
		rm yolo/model/benchmarks.py -f
		rm yolo/model/hubconf.py -f
		rm yolo/model/requirements.txt -f
		rm yolo/model/val.py -f
		rm yolo/model/train.py -f
		rm yolo/model/setup.cfg -f
		rm -rf yolo/model/tmp
		rm -rf yolo/model/weights
		rm -rf yolo/model/segment
		rm -rf yolo/model/data
		rm -rf yolo/model/classify

		# rm -rf yolo/model/runsbackup/$MODELNAME
		# cp -r /home/smile/ai/work/pbs/model/yolov5/runsbackup/$MODELNAME yolo/model/runsbackup/
		# rm yolo/model/runsbackup/$MODELNAME/blood_cell/*.* -f
		# rm yolo/model/runsbackup/$MODELNAME/blood_cell/weights/last.pt -f

		rm -rf yolo/ultralytics/runsbackup/$MODELNAME
		cp -r /home/smile/ai/work/pbs/model/yolov5/runsbackup_v10/$MODELNAME yolo/ultralytics/runsbackup/
		# best.pt 파일을 제외한 blood_cell 폴더 내 모든 파일 삭제
		find yolo/ultralytics/runsbackup/$MODELNAME/blood_cell/ -type f ! -path '*/weights/best.pt' -delete




		ls yolo/model/runsbackup
	elif [ $COMMAND == "del" ]; then
		rm -rf yolo/ultralytics/runsbackup/$MODELNAME
		ls yolo/ultralytics/runsbackup
	elif [ $COMMAND == "list" ]; then
		ls yolo/ultralytics/runsbackup
	elif [ $COMMAND == "push" ]; then
		eval $(ssh-agent)
		ssh-add -t 4w ~/.ssh/smile-bitbucket

		git add .
		git commit -m "모델수정"
		git push
	fi
fi
