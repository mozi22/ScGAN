import os, os.path

prefix = '../dataset_ptb/EvaluationSet/'
postfix = '/rgb'

evaluationfolders = ['bag1','book_turn2','computerbar1','face_turn2','new_ex_occ5_long','static_sign1','toy_wg_occ',
		   'walking_occ_long','basketball1','box_no_occ','computerBar2','flower_red_occ','new_ex_occ6',
		   'studentcenter2.1','toy_wg_occ1','wdog_no1','basketball2','br_occ_0','cup_book','gre_book',
		   'new_ex_occ7.1','studentcenter3.1','toy_yellow_no','wdog_occ3','basketball2.2','br_occ1',
		   'dog_no_1','hand_no_occ','new_student_center1','studentcenter3.2','tracking4','wr_no',
		   'basketballnew','br_occ_turn0','dog_occ_2','hand_occ','new_student_center2','three_people',
		   'tracking7.1','wr_no1','bdog_occ2','cafe_occ1','dog_occ_3','library2.1_occ','new_student_center3',
		   'toy_car_no','two_book','wr_occ2','bear_back','bear_change','bird1.1_no','bird3.1_no','book_move1',
		   'book_turn','cc_occ1','cf_difficult','cf_no_occ','cf_occ2','cf_occ3','child_no2','express1_occ',
		   'express2_occ','express3_static_occ','face_move1','face_occ2','face_occ3','library2.2_occ','mouse_no1',
		   'new_ex_no_occ','new_ex_occ1','new_ex_occ2','new_ex_occ3','new_student_center4','new_student_center_no_occ',
		   'new_ye_no_occ','new_ye_occ','one_book_move','rose1.2','toy_car_occ','toy_green_occ','toy_mo_occ',
		   'toy_no','toy_no_occ','toy_wg_no_occ','two_dog_occ1','two_people_1.1','two_people_1.2','two_people_1.3',
		   'walking_no_occ','walking_occ1','wuguiTwo_no','zball_no1','zball_no2','zball_no3','zballpat_no1']


validationfolders = ['bear_front','child_no1','face_occ5','new_ex_occ4','zcup_move_1']




total_files = 0

for folder in evaluationfolders:
	path = prefix + folder + postfix + '/'
	
	length = len([name for name in os.listdir(path)])

	total_files+=length


print(total_files)