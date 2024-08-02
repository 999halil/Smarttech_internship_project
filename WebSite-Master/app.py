from flask import Flask, flash,render_template, send_file,request, session, redirect, url_for,make_response
from flask import Response,jsonify
import yaml
import cv2
import csv
import random
import PIL 
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash , check_password_hash
from datetime import datetime
import os
from sqlalchemy import text
from jinja2 import Environment
import subprocess
import math
from flask_wtf import FlaskForm
from wtforms import StringField,RadioField,PasswordField,BooleanField,ValidationError, SubmitField, MultipleFileField,IntegerField

from datetime import timedelta
from werkzeug.utils import secure_filename
import uuid as uuid
from flask_wtf import FlaskForm
from wtforms import StringField,RadioField,PasswordField,BooleanField,ValidationError, SubmitField, FloatField
from wtforms.validators import DataRequired, EqualTo, Length , InputRequired

from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user,current_user


from flask import send_from_directory
import sqlite3
#from classes import Notifications,TrainingData,Users,Image,Annotation,Images
from ultralytics import YOLO
from flask import render_template, request
from werkzeug.utils import secure_filename
#from .models import Image, db

# Define jinja2_enumerate function and add it to jinja2 filters
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)

env = Environment()
env.filters['enumerate'] = jinja2_enumerate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.root_path, 'data', 'detection.db')
app.secret_key = "secret_key"
db = SQLAlchemy(app)

################  Classes  ################  

class Users(db.Model, UserMixin):
    id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    
    name = db.Column(db.String(80), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hashed = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(10), nullable=False)

    @property
    def password(self):
        raise AttributeError('Password is not a readable attribute !')
    @password.setter
    def password(self,password):
        self.password_hashed = generate_password_hash(password)

    def verify_password(self,password):
        return check_password_hash(self.password_hashed, password)

    def __repr__(self):
        return '<Name %r>' % self.name

class TrainingData(db.Model):
    __tablename__ = 'TrainingData'
    id = db.Column(db.Integer, primary_key=True)
    epochs = db.Column(db.Integer, nullable=False)
    patience = db.Column(db.Integer, nullable=False)
    batch_size = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    folder_name = db.Column(db.String(100), nullable=False)
    precision = db.Column(db.Float, nullable=False)
    classes_detected= db.Column(db.String(100), nullable=False)
    selected = db.Column(db.String(10), nullable=False)

    def __init__(self, epochs, patience, batch_size, learning_rate,precision,classes_detected):
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classes_detected = classes_detected
        self.precision = precision
        self.created_at = datetime.now()
        self.folder_name = "static/models/" + self.created_at.strftime("%Y-%m-%d_%H-%M-%S") + "/"
        self.selected = "False"

class Notifications(db.Model):
    __tablename__ = 'Notifications'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    message = db.Column(db.String(100), nullable=False)
    frame = db.Column(db.String(1000), nullable=True)

    
    def __init__(self, message, frame):
        self.message = message
        self.frame = frame

class Classes(db.Model):
    __tablename__ = 'Classes'
    id = db.Column(db.Integer, primary_key=True)
    id_class = db.Column(db.Integer)
    name = db.Column(db.String(100), nullable=False)

    def __init__(self, id_class, name):
        self.id_class = id_class
        self.name = name

################  Forms  ################  

class UserForm(FlaskForm):
    name = StringField("Name",validators=[DataRequired()])
    username = StringField("Username",validators=[DataRequired()])
    password_hashed = PasswordField("password",validators=[DataRequired(), EqualTo('passwordh',message='Passwords Must Match!')])
    passwordh = PasswordField("confirm password",validators=[DataRequired()])    
    role = RadioField("Role",validators=[DataRequired()],choices=["Admin","User"])

    submit = SubmitField("Submit")

class ClassForm(FlaskForm):
    id_class = IntegerField("Id_class",validators=[InputRequired()])
    name = StringField("Name",validators=[DataRequired()])

    submit = SubmitField("Submit")

class LoginForm(FlaskForm):
    username = StringField("Username",validators=[DataRequired()])
    password_hashed = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Submit")


class TrainingForm(FlaskForm):
    epochs = FloatField("epochs")
    patience = FloatField("patience")
    batch_size = FloatField("batch size")
    learning_rate = FloatField("learning rate final") 

    submit = SubmitField("Submit")

class UploadDataForm(FlaskForm):
    images = MultipleFileField('Select Images')
    annotations = MultipleFileField('Select Annotations')
    submit = SubmitField('Upload')

################  Routing  ################  

# Flask_Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
	return Users.query.get(int(user_id))


#Create custom Error Pages
#Invalid Url
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


#Internal Server Error
@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500


####### logout Route #######

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
	logout_user()
	flash("You Have Been Logged Out!  Thanks For Stopping By...")
	return redirect(url_for('login'))

####### Home Route #######

@app.route('/')
def home():
    if 'username' in session:
        username = session['username']
        return render_template('home.html', username=username)

    return redirect(url_for('login'))


####### Login Route #######

@app.route('/login', methods=['GET', 'POST'])
def login():
	form = LoginForm()

	if form.validate_on_submit():
		user = Users.query.filter_by(username=form.username.data).first()
		if user:
			# Check the hash
			if check_password_hash(user.password_hashed, form.password_hashed.data):
				login_user(user)
				flash("Login Succesfull!!")
				return redirect(url_for('dashboard'))
			else:
				flash("Wrong Password - Try Again!")
		else:
			flash("That User Doesn't Exist! Try Again...")


	return render_template('login.html', form=form)


####### Add User Route #######

@app.route('/addUserH',methods =['GET','POST'])
@login_required
def addUserH():
    name = None
    
    form = UserForm()
    #validate form
    if form.validate_on_submit():
        user = Users.query.filter_by(username=form.username.data).first()
        if user is None:
            #Hash the password
            hashed_pw =  generate_password_hash(form.password_hashed.data, "pbkdf2:sha256")

            user = Users(name=form.name.data,username=form.username.data,password_hashed=hashed_pw,role=form.role.data)
            db.session.add(user)
            db.session.commit()
            name = form.name.data
            form.name.data = ''
            form.username.data = ''
            form.password_hashed.data = ''
            form.passwordh.data = ''
            form.role.data = ''
            flash("User Added Succesfully !")
        else:
            flash("Username already exists")
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('addUserhashed.html',name=name, form=form)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))

####### Users Route #######

@app.route('/users', methods=['GET'])
@login_required
def users():
    user_list = Users.query.all()
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('users.html', users=user_list)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))

@app.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    user_to_delete = Users.query.filter_by(id=user_id).first()
    db.session.delete(user_to_delete)
    db.session.commit()
    flash(f'{user_to_delete.name} has been deleted.', 'success')
    Admin = current_user.role
    if Admin == "Admin":
        return redirect(url_for('users'))
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))


@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    user_to_edit = Users.query.filter_by(id=user_id).first()

    if request.method == 'POST':
        user_to_edit.username = request.form['username']
        user_to_edit.password_hashed = generate_password_hash(request.form['password']) ####maybe ?
        user_to_edit.role = request.form['role']
        user_to_edit.name = request.form['name']
        db.session.commit()
        flash(f'{user_to_edit.name} has been updated.', 'success')
        return redirect(url_for('users'))

    Admin = current_user.role
    if Admin == "Admin":
        return render_template('edit_user.html', user=user_to_edit)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))
    
    
####### classes Route #######

@app.route('/classes', methods=['GET'])
@login_required
def classes():
    class_list = Classes.query.all()
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('classes.html', classes=class_list)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))


@app.route('/classes/<int:class_idd>/delete', methods=['POST'])
@login_required
def delete_class(class_idd):
    class_to_delete = Classes.query.filter_by(id=class_idd).first()
    db.session.delete(class_to_delete)
    db.session.commit()
    flash(f'{class_to_delete.name} has been deleted.', 'success')
    Admin = current_user.role
    if Admin == "Admin":
        return redirect(url_for('classes'))
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))
    

@app.route('/addClass',methods =['GET','POST'])
@login_required
def addClass():
    form = ClassForm()
    #validate form
    if form.validate_on_submit():
        classs = Classes.query.filter_by(id_class=form.id_class.data).first()
        if classs is None:

            classs = Classes(name=form.name.data,id_class=form.id_class.data)
            db.session.add(classs)
            db.session.commit()

            flash("Class Added Succesfully !")
        else:
            flash("Class already exists")
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('addclass.html', form=form)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))



####### Dashboard Route #######

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/webcam')
def webcam():
     return Response(generate_frames(path_x=0),mimetype='multipart/x-mixed-replace; boundary=frame')

####### Train Route #######

def copy_files_with_names(source_folder, destination_folder, file_names):
    for file_name in file_names:

        source_path = os.path.join(source_folder, file_name)
        source_path = source_path.replace("\\", "/")

        destination_path = os.path.join(destination_folder, file_name)
        destination_path = destination_path.replace("\\", "/")

        shutil.copyfile(source_path, destination_path)



def copy_file_with_name(source_folder, destination_folder, file_name):
    source_path = os.path.join(source_folder, file_name)
    source_path = source_path.replace("\\", "/")
    
    destination_path = os.path.join(destination_folder, file_name)
    destination_path = destination_path.replace("\\", "/")
    # Copy the file
    shutil.copyfile(source_path, destination_path)


def copy_to_train_val(percentageT):
    # define the source folder path where the files are located
    source_flder_path = 'datasets/all/images/'

    # define the destination folder names
    destination_train = "train/"
    destination_val = "val/"
    # destination_test = "test"
    # percentageT = 70
    # percentageV = 10
    # create the destination folders if they don't exist
    destination_train = os.path.join('datasets/', destination_train)
    destination_val = os.path.join('datasets/', destination_val)
    # destination_test = os.path.join('/content/drive/MyDrive/oil/', destination_test)
    if not os.path.exists(destination_train):
        os.mkdir(destination_train)
    if not os.path.exists(destination_val):
        os.mkdir(destination_val)
    #if not os.path.exists(destination_test):
    #    os.mkdir(destination_test)
    # get a list of all files in the source folder
    all_files = os.listdir(source_flder_path)
    # calculate the number of files to copy to each destination folder based on the percentages
    num_filesTrain = int(len(all_files) * (percentageT / 100)) #35
    num_filesVal=int(len(all_files)-num_filesTrain)
    print(num_filesTrain)
    print(num_filesVal)

    # randomly select the files to copy to each destination folder
    files1 = random.sample(all_files, num_filesTrain)
    files2 = random.sample(list(set(all_files) - set(files1)), num_filesVal)
    # copy the files to the train folder
    for filename in files1:
        source_pfile_path = os.path.join('datasets/all/images/', filename)
        destination_pfile_path = os.path.join('datasets/train/images/', filename)
        
        print(source_pfile_path)
        print(destination_pfile_path)
        shutil.copy2(source_pfile_path,destination_pfile_path)
    # copy the files to the test folder  
    for filename in files2:
        source_pfile_path = os.path.join('datasets/all/images/', filename)
        destination_pfile_path = os.path.join('datasets/val/images/', filename)
        shutil.copy2(source_pfile_path, destination_pfile_path)
    # define the folder path where the .jpg and .txt files are located
    #########################for the training folder#################################""
    folder_pathJ = "datasets/train/images/"
    folder_pathT = "datasets/all/labels/"
    folder_pathNT = "datasets/train/labels/"

    jpg_files = [f for f in os.listdir(folder_pathJ) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # get the base name of the .jpg file (i.e., without the extension)
        base_name = os.path.splitext(jpg_file)[0]
    
        # check if there's a corresponding .txt file with the same name
        txt_file = base_name + '.txt'
        if txt_file in os.listdir(folder_pathT):
            # copy the .txt file to a new folder (in this case, the same folder)
            shutil.copy2(os.path.join(folder_pathT, txt_file), os.path.join(folder_pathNT, txt_file))
    # define the folder path where the .jpg and .txt files are located
    #########################for the test folder#################################""
    folder_pathJ = "datasets/val/images/"
    folder_pathT = "datasets/all/labels/"
    folder_pathNT = "datasets/val/labels/"

    jpg_files = [f for f in os.listdir(folder_pathJ) if f.endswith('.jpg')]
    for jpg_file in jpg_files:
        # get the base name of the .jpg file (i.e., without the extension)
        base_name = os.path.splitext(jpg_file)[0]
    
        # check if there's a corresponding .txt file with the same name
        txt_file = base_name + '.txt'
        if txt_file in os.listdir(folder_pathT):
            # copy the .txt file to a new folder (in this case, the same folder)
            shutil.copy2(os.path.join(folder_pathT, txt_file), os.path.join(folder_pathNT, txt_file))
    

def find_max_value(csv_file, column_name):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        max_value = None
        for row in reader:
            value = float(row[column_name])  # Assuming the values in the column are numeric
            if max_value is None or value > max_value:
                max_value = value
        return max_value

def modify_custom_yaml(names):
    with open('data_custom.yaml', 'r') as file:
        data = yaml.safe_load(file)

    data['names'] = {i: name for i, name in enumerate(names)}

    with open('data_custom.yaml', 'w') as file:
        yaml.dump(data, file)

@app.route('/train', methods=['GET', 'POST'])
@login_required
def train():
    form = TrainingForm()

    ep=0
    ep=form.epochs.data
    #validate form
    if form.validate_on_submit():

        source_folder = 'runs/detect/train'
        if os.path.exists(source_folder):
            shutil.rmtree(source_folder)

        model = YOLO('yolov8n.pt')
        print("model downloaded")
        if not form.epochs.data:
            t_epochs= 100 
        else:
            t_epochs=int(form.epochs.data)

        if not form.batch_size.data:
            t_batch=16
        else:
            t_batch=int(form.batch_size.data) 

        if not form.patience.data:
            t_patience=50
        else:
            t_patience=int(form.patience.data) 

        if not form.batch_size.data:
            t_lrf=0.01
        else:
            t_lrf=float(form.learning_rate.data)
        copy_to_train_val(70)

        modify_custom_yaml()

        model.train(data='data_custom.yaml', epochs=t_epochs, batch=t_batch, lrf=t_lrf, patience=t_patience,imgsz=640)

        csv_file = 'runs/detect/train/results.csv'
        column_name = '    metrics/mAP50-95(B)'  # Replace with the actual column name in your CSV file
        max_value = find_max_value(csv_file, column_name)

        delete_files_in_folder("datasets/train/images")
        delete_files_in_folder("datasets/train/labels")
        delete_files_in_folder("datasets/val/images")
        delete_files_in_folder("datasets/val/labels")

        existing_instances = Classes.query.all()
        instance_names = [instance.name for instance in existing_instances]
        names_string = ', '.join(instance_names)

        
        training = TrainingData(epochs=t_epochs,batch_size=t_batch,patience=form.patience.data,learning_rate=t_lrf,precision=max_value,classes_detected=names_string)
        db.session.add(training)
        db.session.commit()

        subfolder_path = training.folder_name  
              
        os.makedirs(subfolder_path)

        

        file_names = ['P_curve.png', 'PR_curve.png', 'R_curve.png','results.png','results.csv']
        copy_files_with_names(source_folder, subfolder_path, file_names)
        
        source_folder2 = 'runs/detect/train/weights'
        file_name = 'best.pt'
        copy_file_with_name(source_folder2, subfolder_path,file_name)

        #shutil.rmtree(source_folder)

        print("Training.")
        print("Training..")
        print("Training...")
        
        form.epochs.data = ''
        form.batch_size.data = ''
        form.patience.data = ''
        form.learning_rate.data = ''
        flash("Trained Succesfully !")

    Admin = current_user.role
    if Admin == "Admin":
        return render_template('trainForm.html', form=form,ep=ep)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))
    

####### notifications Route #######

@app.route('/notifications', methods=['GET'])
@login_required
def notifications():
    notif_list = Notifications.query.all()
    return render_template('notifications.html', notifications=notif_list)

####### notifications 1 #######
@app.route('/notifications/<int:notification_id>/addtoDataset_notificationFP', methods=['POST'])
@login_required
def addtoDataset_notificationFP(notification_id):

    image_to_add = Notifications.query.filter_by(id=notification_id).first()

    source_image_file = image_to_add.frame.replace("images","raw_images")

    destination_image_file = source_image_file.replace("notifications/raw_images/","updateDatasetFile/image/")

    shutil.copy2(source_image_file, destination_image_file) #raw_ images to updateDatasetFile
    
    if os.path.exists(source_image_file):
        os.remove(source_image_file) #notifications > raw_images
    print(source_image_file)
    raw_notif_annotation=source_image_file.replace("raw_images","raw_annotations")
    raw_notif_annotation=raw_notif_annotation.replace(".jpg",".txt")

    if os.path.exists(raw_notif_annotation): #notifications > raw_annotations
        os.remove(raw_notif_annotation)
    print(raw_notif_annotation)
    if os.path.exists(image_to_add.frame):
        os.remove(image_to_add.frame) #notifications > images
    print(image_to_add.frame)

    upload_folder = "static/DataSetimages"
    Dataset_image_folder ="datasets/all/images/"
    image_folder ="static/updateDatasetFile/image/"

        
    copy_files_in_folder(image_folder,Dataset_image_folder) #raw updateDatasetFile to all
    copy_files_in_folder(image_folder,upload_folder) #raw updateDatasetFile to DataSetimages
    delete_files_in_folder(image_folder) #delete updateDatasetFile
    
    db.session.delete(image_to_add)
    db.session.commit()
    flash(f'{image_to_add.timestamp} has been added.', 'success')
    return redirect(url_for('notifications'))


####### notifications 2 #######
@app.route('/notifications/<int:notification_id>/addtoDataset_notification', methods=['POST'])
@login_required
def addtoDataset_notification(notification_id):

    image_to_add = Notifications.query.filter_by(id=notification_id).first()

    source_image_file = image_to_add.frame.replace("images","raw_images")
    
    source_image_annotation=source_image_file.replace("raw_images","raw_annotations")
    source_image_annotation=source_image_annotation.replace(".jpg",".txt")

    destination_image_file = source_image_file.replace("notifications/raw_images/","updateDatasetFile/image/")

    destination_annotation_file = source_image_annotation.replace("notifications/raw_annotations/","updateDatasetFile/label/")

    shutil.copy2(source_image_file, destination_image_file)
    shutil.copy2(source_image_annotation, destination_annotation_file)

    if os.path.exists(source_image_file):
        os.remove(source_image_file) #notifications > raw_images
    print(source_image_file)
    raw_notif_annotation=source_image_file.replace("raw_images","raw_annotations")
    raw_notif_annotation=raw_notif_annotation.replace(".jpg",".txt")

    if os.path.exists(raw_notif_annotation): #notifications > raw_annotations
        os.remove(raw_notif_annotation)
    print(raw_notif_annotation)
    if os.path.exists(image_to_add.frame):
        os.remove(image_to_add.frame) #notifications > images
    print(image_to_add.frame)


    upload_folder = "static/DataSetimages"

    Dataset_image_folder ="datasets/all/images/"
    Dataset_label_folder ="datasets/all/labels/"

    image_folder ="static/updateDatasetFile/image/"
    label_folder ="static/updateDatasetFile/label/"

    class_objects = Classes.query.all()

    # Extract the names from each object
    class_names = [obj.name for obj in class_objects]

    process_images(image_folder,label_folder,upload_folder,class_names)
        
    copy_files_in_folder(image_folder,Dataset_image_folder)
    copy_files_in_folder(label_folder,Dataset_label_folder)
    delete_files_in_folder(image_folder)
    delete_files_in_folder(label_folder)
    
    db.session.delete(image_to_add)
    db.session.commit()
    flash(f'{image_to_add.timestamp} has been added.', 'success')
    return redirect(url_for('notifications'))



####### notifications 3 #######
@app.route('/notifications/<int:notification_id>/delete_notification', methods=['POST'])
@login_required
def delete_notification(notification_id):
    notification_to_delete = Notifications.query.filter_by(id=notification_id).first()
    raw_notif=notification_to_delete.frame.replace("images","raw_images")
    
    if os.path.exists(raw_notif):
        os.remove(raw_notif)
    
    raw_notif_annotation=raw_notif.replace("raw_images","raw_annotations")
    raw_notif_annotation=raw_notif_annotation.replace(".jpg",".txt")

    if os.path.exists(raw_notif_annotation):
        os.remove(raw_notif_annotation)

    if os.path.exists(notification_to_delete.frame):
        os.remove(notification_to_delete.frame)

    db.session.delete(notification_to_delete)
    db.session.commit()
    flash(f'{notification_to_delete.timestamp} has been deleted.', 'success')
    return redirect(url_for('notifications'))

####### notifications 4 #######
@app.route('/notifications/<int:notification_id>/download_notif', methods=['GET','POST'])
@login_required
def download_notif(notification_id):
    if request.method == 'POST' or 'GET':
        delete_files_in_folder('uploads/')
        image_to_add = Notifications.query.filter_by(id=notification_id).first()
    
        source_image_file = image_to_add.frame.replace("images","raw_images")
        print(source_image_file)
        # Split the directory into the path and the filename
        image_path, image_name = os.path.split(source_image_file)

        uploads_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
        #copy_file_in_folder(image_path,"toDownload/",image_name)

        
        destination_image_file = source_image_file.replace("static/notifications/raw_images/","uploads/")
        print(destination_image_file)
        shutil.copy2(source_image_file, destination_image_file)

        response = make_response(send_from_directory(directory=uploads_folder, path=image_name, as_attachment=True))

        response.headers['Content-Disposition'] = f'attachment; filename={image_name}'

        raw_notif=image_to_add.frame.replace("images","raw_images")
    
        if os.path.exists(raw_notif):
            os.remove(raw_notif)
    
        raw_notif_annotation=raw_notif.replace("raw_images","raw_annotations")
        raw_notif_annotation=raw_notif_annotation.replace(".jpg",".txt")

        if os.path.exists(raw_notif_annotation):
            os.remove(raw_notif_annotation)

        if os.path.exists(image_to_add.frame):
            os.remove(image_to_add.frame)

        db.session.delete(image_to_add)
        db.session.commit()
        
        flash(f'{image_to_add.timestamp} has been downloaded.', 'success')
        return response 
    
    return redirect(url_for('notifications'))
    
    

####### Upload Route #######
IMAGES_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in IMAGES_EXTENSIONS


@app.route('/uploadmulti', methods=['GET', 'POST'])
@login_required
def uploadmulti():
    form = UploadDataForm()
    if form.validate_on_submit():
        images = form.images.data
        annotations = form.annotations.data

        image_folder ="static/updateDataset/images/"
        label_folder ="static/updateDataset/labels/"

        for image in images:
            if image and allowed_image(image.filename):
                imagename = secure_filename(image.filename)
                image.save(os.path.join(image_folder, imagename))

        for annotation in annotations:
            if annotation and annotation.filename.endswith('.txt'):
                filename = secure_filename(annotation.filename)
                annotation.save(os.path.join(label_folder, filename))

        upload_folder = "static/DataSetimages"
        '''
        new_size = (640, 640)

        for filename in os.listdir(image_folder):
            if filename.endswith('.jpg'):
        
                image = PIL.Image.open(os.path.join(image_folder, filename))

                resized_image = image.resize(new_size)

                #new_filename = 'resized_' + filename
                resized_image.save(os.path.join(image_folder, filename))
'''
        Dataset_image_folder ="datasets/all/images/"
        Dataset_label_folder ="datasets/all/labels/"

        class_objects = Classes.query.all()

        # Extract the names from each object
        class_names = [obj.name for obj in class_objects]


        process_images(image_folder,label_folder,upload_folder,class_names)
        
        copy_files_in_folder(image_folder,Dataset_image_folder)
        copy_files_in_folder(label_folder,Dataset_label_folder)
        delete_files_in_folder(image_folder)
        delete_files_in_folder(label_folder)
        
        flash('Files uploaded successfully!') 
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('addtoDataset.html', form=form)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))

####### Dataset Route #######

@app.route('/dataset', methods=['GET'])
@login_required
def dataset():
    img_folder = "static/DataSetimages/"
    images = os.listdir(img_folder)
    image_urls = [os.path.join(img_folder,image)for image in images]
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('dataset.html', image_urls=image_urls)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))
    

@app.route('/delete_dataset_images', methods=['POST'])
@login_required
def delete_dataset_images():
    delete_files_in_folder('static/DataSetimages/')
    delete_files_in_folder('datasets/all/images/')
    delete_files_in_folder('datasets/all/labels/')
    Admin = current_user.role
    if Admin == "Admin":
        return redirect(url_for('dataset'))
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))



def process_images(image_folder, label_folder, output_folder,className):
    # List all image files and label files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through image files
    for image_file in image_files:
        # Find the corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file in label_files:
            # Read the label file and extract annotations
            label_file_path = os.path.join(label_folder, label_file)
            annotations = extract_variables(label_file_path)

            # Read the image file
            image_file_path = os.path.join(image_folder, image_file)
            image = PIL.Image.open(image_file_path)
            
            # Call the transform_to_annotated_img function
            new_image = transform_to_annotated_img(image, annotations, className)

            # Save the new annotated image to the output folder
            new_image_file_path = os.path.join(output_folder, image_file)
            cv2.imwrite(new_image_file_path, new_image)

def process_images_withoutlabel(image_folder, output_folder):
    # List all image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate through image files
    for image_file in image_files:
        # Read the image file
        image_file_path = os.path.join(image_folder, image_file)
        image = PIL.Image.open(image_file_path)
        # Save the new annotated image to the output folder
        new_image_file_path = os.path.join(output_folder, image_file)
        cv2.imwrite(new_image_file_path, image)

def delete_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
 
import shutil
def copy_files_in_folder(source_folder,destination_folder):
    file_list = os.listdir(source_folder)
    # Iterate through the files and copy them to the destination folder
    for file_name in file_list:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy2(source_file, destination_file)

def copy_file_in_folder(source_folder,destination_folder,file_name):
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    shutil.copy2(source_file, destination_file)

def extract_variables(file_path):
    variables = []
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming the variables are space-separated
            parts = line.strip().split(' ')
            if len(parts) == 5:
                variables.append(parts)
    return variables



def transform_to_annotated_img(image,annotation,className):
    img_width, img_height = image.size
    new_image = image.copy()
    
    for line in annotation:
        class_label, x_center, y_center, width, height = line
        x_center = float(x_center) * img_width
        y_center = float(y_center) * img_height
        width = float(width) * img_width
        height = float(height) * img_height

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        new_image = np.array(new_image)
        # Convert the image from RGB to BGR format
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        # Draw the rectangle and text
        cv2.rectangle(new_image, (x_min, y_min), (x_max, y_max), (255,0,255), 3)
        cv2.putText(new_image, className[int(class_label)], (x_min, y_min - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 3)
        
        
    return new_image
     
     

def video_detect(img,classNames):
    model = YOLO("models/model_used/best.pt")

    # Detect oil leaks in frame
    # Initialize variables
    oil_leak_detected = False
    oil_leak_count = 0
    results=model(img,stream=True)
    image = PIL.Image.fromarray(img)

    img_width, img_height = image.size
    annotation = [0,0,0,0,0]
    for r in results:
            boxes=r.boxes
            for box in boxes:
                conf=math.ceil((box.conf[0]*100))/100
                #if confidence score is high
                if conf>0.99:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    print(x1,y1,x2,y2)

                    cls=int(box.cls[0])
                    class_name=classNames[cls]
                    label=f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    #print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3

                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    print("cls: " + str(cls) + "x_center " + str(x_center) + "y_center " + str(y_center) + "width " + str(width) + "height " + str(height)) 
                    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                    cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
                    annotation = [cls,x_center,y_center,width,height]
                    oil_leak_detected = True
                
    return img, oil_leak_detected , annotation


def video_detection(path_x):
    with app.app_context():
        video_capture = path_x
        #create webcam object
        cap=cv2.VideoCapture(video_capture)
        frame_width=int(cap.get(3))
        frame_width=int(cap.get(4))
    
        while True:
            success, img = cap.read()
            # Check if frame was read successfully
            if not success:
               break
            
            image_clear=img.copy() #maghir cadre
            # Detect oil leaks in frame
            class_objects = Classes.query.all()

            # Extract the names from each object
            classNames = [obj.name for obj in class_objects]
            img, oil_leak_detected,annotation = video_detect(img,classNames)
            print(oil_leak_detected)
            if oil_leak_detected ==True:
            
                current_time = datetime.now().strftime("%Y-%m-%d__%H_%M_%S_%f")[:-1]
            #grab image name
                relative_path = "static/notifications/images/"
                img_name = relative_path + str(current_time) + ".jpg" #+ str(uuid.uuid1()) + '.jpg'
            #save the frame
            
            #change it to a string to save to db
            

                msg = "Detection on Camera 1"
            # Create blob from frame
            #blob = cv2.dnn.blobFromImage(img, 1/255, (640, 640), swapRB=True, crop=False)
                notif = Notifications.query.order_by(Notifications.id.desc()).first()
                if not notif:
                     notifBool = False
                else:
                    #notif_timestamp = datetime.strptime(notif.timestamp, '%Y-%m-%d %H:%M:%S.%f')
                    
                    notifBool = (notif.timestamp >= (datetime.now() - timedelta(minutes=1)) )

                if notifBool == False:
                
                    
                    print("current time is " + current_time)
                    print(msg)
                    print(img_name)
                    
                    img_name2 = img_name.replace("images","raw_images")

                    file_name = img_name2.replace("raw_images","raw_annotations")
                    file_name = file_name.replace(".jpg",".txt")
                    print(len(annotation))

                    print("annotation cls: " + str(annotation[0]) + "x_center " + str(annotation[1]) + "y_center " + str(annotation[2]) + "width " + str(annotation[3]) + "height ") # + str(annotation[4])
                    
                    with open(file_name,"w") as file:
                        
                        for num in annotation:
                            float_str = str(num)
                            file.write(float_str)
                            file.write(" ")
                    file.close()

                    img_name2 = img_name.replace("images","raw_images")
                    cv2.imwrite( img_name2 , image_clear)
                    cv2.imwrite( img_name , img)
                    notification = Notifications(message=msg,frame=img_name)
                    db.session.add(notification)
                    db.session.commit()
                    
            yield img            

#from predict import video_detection

def generate_frames(path_x =''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
####### Models Route #######

@app.route('/models', methods=['GET'])
@login_required
def models():
    model_list = TrainingData.query.all()
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('models.html', models=model_list)
    else:
        flash("sorry you must be an Admin to access this page")
        return redirect(url_for('dashboard'))
    

####### settings Route #######
def reset_selected():
    all_data = TrainingData.query.all()
    for data in all_data:
        data.selected = "False"
    db.session.commit()


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    Admin = current_user.role


    if request.method == 'POST':

        # Perform necessary actions with selected_model
        new_model_folder = request.form.get('model')
        
        selected_model = TrainingData.query.filter_by(folder_name=new_model_folder).first()
        reset_selected()
        selected_model.selected = "True"
        db.session.commit()

        detection_model_folder = "models/model_used/"
        selected_model_folder = selected_model.folder_name 

        delete_files_in_folder(detection_model_folder)
        copy_file_in_folder(selected_model_folder,detection_model_folder,"best.pt")

        class_list = Classes.query.all()
        

        return redirect(url_for('settings'))
    else:
        model_list = TrainingData.query.all()
        if Admin == "Admin":
            return render_template('settings.html', models=model_list)
        else:
            flash("sorry you must be an Admin to access this page")
            return redirect(url_for('dashboard'))





if __name__ == '__main__':
    
    #model = YOLO('/models/yolov8n.pt') 

    app.run(debug=True)
