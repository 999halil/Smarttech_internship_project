# Assuming the YOLOv10 package is named 'ultralytics_v10' or similar
from flask_wtf.csrf import generate_csrf
from flask import Flask, flash, render_template, send_file, request, session, redirect, url_for, make_response, Response, jsonify
import yaml
import cv2
import csv
import random
import PIL
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
from sqlalchemy import text
from jinja2 import Environment
import subprocess
import math
from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, PasswordField, BooleanField, ValidationError, SubmitField, MultipleFileField, IntegerField, FloatField
from wtforms.validators import DataRequired, EqualTo, InputRequired, Length
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import uuid as uuid
import shutil
from ultralytics import YOLO
from flask import send_from_directory
from flask_cors import CORS
from flask_wtf import CSRFProtect

# Define jinja2_enumerate function and add it to jinja2 filters


def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)


env = Environment()
env.filters['enumerate'] = jinja2_enumerate

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(app.root_path, 'data', 'detection.db')
app.secret_key = "secret_key"
# csrf = CSRFProtect(app)

db = SQLAlchemy(app)
CORS(app)


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
    def password(self, password):
        self.password_hashed = generate_password_hash(password)

    def verify_password(self, password):
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
    classes_detected = db.Column(db.String(100), nullable=False)
    selected = db.Column(db.String(10), nullable=False)

    def __init__(self, epochs, patience, batch_size, learning_rate, precision, classes_detected):
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.classes_detected = classes_detected
        self.precision = precision
        self.created_at = datetime.now()
        self.folder_name = "static/models/" + \
            self.created_at.strftime("%Y-%m-%d_%H-%M-%S") + "/"
        self.selected = "False"

    def to_dict(self):
        return {
            "id": self.id,
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "folder_name": self.folder_name,
            "precision": self.precision,
            "classes_detected": self.classes_detected,
            "selected": self.selected
        }


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

    def to_dict(self):
        return {
            'id': self.id,
            'id_class': self.id_class,
            'name': self.name
        }


################  Forms  ################


class UserForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    username = StringField("Username", validators=[DataRequired()])
    password_hashed = PasswordField("password", validators=[
                                    DataRequired(), EqualTo('passwordh', message='Passwords Must Match!')])
    passwordh = PasswordField("confirm password", validators=[DataRequired()])
    role = RadioField("Role", validators=[
                      DataRequired()], choices=["Admin", "User"])
    submit = SubmitField("Submit")


class ClassForm(FlaskForm):
    id_class = IntegerField("Id_class", validators=[InputRequired()])
    name = StringField("Name", validators=[DataRequired()])
    submit = SubmitField("Submit")


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password_hashed = PasswordField("Password", validators=[DataRequired()])
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


def create_tables():
    db.create_all()
    # Ensure at least one setting is created
    if not GeneralSettings.query.first():
        setting = GeneralSettings(confidence=0.5)
        db.session.add(setting)
        db.session.commit()
################  Routing  ################


# Flask_Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

# Create custom Error Pages
# Invalid Url


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

# Internal Server Error


@app.errorhandler(500)
def page_not_found(e):
    return render_template("500.html"), 500

####### logout Route #######


@app.route('/logout', methods=['GET', 'POST'])
# login_required
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
                flash("Login Successful!!")
                return redirect(url_for('dashboard'))
            else:
                flash("Wrong Password - Try Again!")
        else:
            flash("That User Doesn't Exist! Try Again...")
    return render_template('login.html', form=form)

####### Add User Route #######


@app.route('/addUserH', methods=['GET', 'POST'])
# login_required
def addUserH():
    name = None
    form = UserForm()
    # Validate form
    if form.validate_on_submit():
        user = Users.query.filter_by(username=form.username.data).first()
        if user is None:
            # Hash the password
            hashed_pw = generate_password_hash(
                form.password_hashed.data, "pbkdf2:sha256")
            user = Users(name=form.name.data, username=form.username.data,
                         password_hashed=hashed_pw, role=form.role.data)
            db.session.add(user)
            db.session.commit()
            name = form.name.data
            form.name.data = ''
            form.username.data = ''
            form.password_hashed.data = ''
            form.passwordh.data = ''
            form.role.data = ''
            flash("User Added Successfully!")
        else:
            flash("Username already exists")
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('addUserhashed.html', name=name, form=form)
    else:
        flash("Sorry, you must be an Admin to access this page")
        return redirect(url_for('dashboard'))

####### Users Route #######


@app.route('/users', methods=['GET'])
# login_required
def users():
    user_list = Users.query.all()
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('users.html', users=user_list)
    else:
        flash("Sorry, you must be an Admin to access this page")
        return redirect(url_for('dashboard'))


@app.route('/users/<int:user_id>/delete', methods=['POST'])
# login_required
def delete_user(user_id):
    user_to_delete = Users.query.filter_by(id=user_id).first()
    db.session.delete(user_to_delete)
    db.session.commit()
    flash(f'{user_to_delete.name} has been deleted.', 'success')
    Admin = current_user.role
    if Admin == "Admin":
        return redirect(url_for('users'))
    else:
        flash("Sorry, you must be an Admin to access this page")
        return redirect(url_for('dashboard'))


@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
# login_required
def edit_user(user_id):
    user_to_edit = Users.query.filter_by(id=user_id).first()
    if request.method == 'POST':
        user_to_edit.username = request.form['username']
        user_to_edit.password_hashed = generate_password_hash(
            request.form['password'])
        user_to_edit.role = request.form['role']
        user_to_edit.name = request.form['name']
        db.session.commit()
        flash(f'{user_to_edit.name} has been updated.', 'success')
        return redirect(url_for('users'))
    Admin = current_user.role
    if Admin == "Admin":
        return render_template('edit_user.html', user=user_to_edit)
    else:
        flash("Sorry, you must be an Admin to access this page")
        return redirect(url_for('dashboard'))

####### Classes Route #######


@app.route('/api/classes', methods=['GET'])
def get_classes():
    classes = Classes.query.all()
    return jsonify([cls.to_dict() for cls in classes]), 200


@app.route('/api/classes', methods=['POST'])
def add_class():
    data = request.get_json()
    new_class = Classes(id_class=data['id_class'], name=data['name'])
    db.session.add(new_class)
    db.session.commit()
    return jsonify(new_class.to_dict()), 201


@app.route('/api/classes/<int:id>', methods=['PUT'])
def update_class(id):
    data = request.get_json()
    cls = Classes.query.get(id)
    if not cls:
        return jsonify({'error': 'Class not found'}), 404
    cls.id_class = data['id_class']
    cls.name = data['name']
    db.session.commit()
    return jsonify(cls.to_dict()), 200


@app.route('/api/classes/<int:id>', methods=['DELETE'])
def delete_class(id):
    cls = Classes.query.get(id)
    if not cls:
        return jsonify({'error': 'Class not found'}), 404
    db.session.delete(cls)
    db.session.commit()
    return '', 204

####### Dashboard Route #######


@app.route('/dashboard', methods=['GET', 'POST'])
# login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    shutil.copyfile(source_path, destination_path)


def copy_to_train_val(percentageT):
    source_flder_path = 'datasets/all/images/'
    destination_train = "train/"
    destination_val = "val/"
    destination_train = os.path.join('datasets/', destination_train)
    destination_val = os.path.join('datasets/', destination_val)
    if not os.path.exists(destination_train):
        os.mkdir(destination_train)
    if not os.path.exists(destination_val):
        os.mkdir(destination_val)
    all_files = os.listdir(source_flder_path)
    num_filesTrain = int(len(all_files) * (percentageT / 100))
    num_filesVal = int(len(all_files) - num_filesTrain)
    print(num_filesTrain)
    print(num_filesVal)
    files1 = random.sample(all_files, num_filesTrain)
    files2 = random.sample(list(set(all_files) - set(files1)), num_filesVal)
    for filename in files1:
        source_pfile_path = os.path.join('datasets/all/images/', filename)
        destination_pfile_path = os.path.join(
            'datasets/train/images/', filename)
        print(source_pfile_path)
        print(destination_pfile_path)
        shutil.copy2(source_pfile_path, destination_pfile_path)
    for filename in files2:
        source_pfile_path = os.path.join('datasets/all/images/', filename)
        destination_pfile_path = os.path.join('datasets/val/images/', filename)
        shutil.copy2(source_pfile_path, destination_pfile_path)
    folder_pathJ = "datasets/train/images/"
    folder_pathT = "datasets/all/labels/"
    folder_pathNT = "datasets/train/labels/"
    jpg_files = [f for f in os.listdir(folder_pathJ) if f.endswith('.jpg')]
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        txt_file = base_name + '.txt'
        if txt_file in os.listdir(folder_pathT):
            shutil.copy2(os.path.join(folder_pathT, txt_file),
                         os.path.join(folder_pathNT, txt_file))
    folder_pathJ = "datasets/val/images/"
    folder_pathT = "datasets/all/labels/"
    folder_pathNT = "datasets/val/labels/"
    jpg_files = [f for f in os.listdir(folder_pathJ) if f.endswith('.jpg')]
    for jpg_file in jpg_files:
        base_name = os.path.splitext(jpg_file)[0]
        txt_file = base_name + '.txt'
        if txt_file in os.listdir(folder_pathT):
            shutil.copy2(os.path.join(folder_pathT, txt_file),
                         os.path.join(folder_pathNT, txt_file))


def find_max_value(csv_file, column_name):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        # Strip spaces from field names
        reader.fieldnames = [field.strip() for field in reader.fieldnames]

        if column_name not in reader.fieldnames:
            raise KeyError(
                f"Column '{column_name}' not found in CSV file headers: {reader.fieldnames}")

        max_value = None
        for row in reader:
            try:
                value = float(row[column_name])
                if max_value is None or value > max_value:
                    max_value = value
            except KeyError:
                print(
                    f"Warning: Row does not contain the column '{column_name}'. Row: {row}")
            except ValueError:
                print(
                    f"Warning: Value in column '{column_name}' could not be converted to float. Value: {row[column_name]}")
        return max_value


def modify_custom_yaml(names):
    with open('data_custom.yaml', 'r') as file:
        data = yaml.safe_load(file)
    data['names'] = {i: name for i, name in enumerate(names)}
    with open('data_custom.yaml', 'w') as file:
        yaml.dump(data, file)


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()

    # Extract and validate the training parameters from the JSON payload
    try:
        epochs = int(data.get('epochs', 100))
        batch_size = int(data.get('batch_size', 16))
        patience = int(data.get('patience', 50))
        learning_rate = float(data.get('learning_rate', 0.01))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input data"}), 400

    # Ensure directories exist or create them
    source_folder = 'runs/detect/train'
    if os.path.exists(source_folder):
        shutil.rmtree(source_folder)

    # Load the YOLO model
    model = YOLO('yolov10n.pt')
    print("Model downloaded")

    # Perform training
    try:
        # Define `copy_to_train_val` and other helper functions as needed
        # Assuming this function exists and is defined properly
        copy_to_train_val(70)
        model.train(data='data_custom.yaml', epochs=epochs,
                    batch=batch_size, lrf=learning_rate, patience=patience, imgsz=640)

        # Handle post-training tasks
        csv_file = 'runs/detect/train/results.csv'
        column_name = 'metrics/mAP50-95(B)'
        # Assuming function is defined
        max_value = find_max_value(csv_file, column_name)

        # Assuming the functions for cleanup and storing training results are defined
        delete_files_in_folder("datasets/train/images")
        delete_files_in_folder("datasets/train/labels")
        delete_files_in_folder("datasets/val/images")
        delete_files_in_folder("datasets/val/labels")

        # Get existing classes and names
        existing_instances = Classes.query.all()
        instance_names = [instance.name for instance in existing_instances]
        names_string = ', '.join(instance_names)

        # Store the training result in the database
        training = TrainingData(epochs=epochs, batch_size=batch_size, patience=patience,
                                learning_rate=learning_rate, precision=max_value, classes_detected=names_string)
        db.session.add(training)
        db.session.commit()

        # Handle model saving and file copying
        subfolder_path = training.folder_name
        os.makedirs(subfolder_path)
        file_names = ['P_curve.png', 'PR_curve.png',
                      'R_curve.png', 'results.png', 'results.csv']
        copy_files_with_names(source_folder, subfolder_path, file_names)
        source_folder2 = 'runs/detect/train/weights'
        file_name = 'best.pt'
        copy_file_with_name(source_folder2, subfolder_path, file_name)

        return jsonify({"message": "Training started successfully!"}), 200

    except Exception as e:
        # Log the exception and return an error response
        print(f"Training error: {str(e)}")
        return jsonify({"error": "Training failed due to an internal error"}), 500


@app.route('/train', methods=['GET'])
# @login_required
def trainget():
    form = TrainingForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            source_folder = 'runs/detect/train'
            if os.path.exists(source_folder):
                shutil.rmtree(source_folder)
            model = YOLO('yolov10n.pt')
            print("model downloaded")
            t_epochs = int(form.epochs.data) if form.epochs.data else 100
            t_batch = int(form.batch_size.data) if form.batch_size.data else 16
            t_patience = int(form.patience.data) if form.patience.data else 50
            t_lrf = float(
                form.learning_rate.data) if form.learning_rate.data else 0.01
            copy_to_train_val(70)
            model.train(data='data_custom.yaml', epochs=t_epochs,
                        batch=t_batch, lrf=t_lrf, patience=t_patience, imgsz=640)
            csv_file = 'runs/detect/train/results.csv'
            column_name = 'metrics/mAP50-95(B)'
            max_value = find_max_value(csv_file, column_name)
            delete_files_in_folder("datasets/train/images")
            delete_files_in_folder("datasets/train/labels")
            delete_files_in_folder("datasets/val/images")
            delete_files_in_folder("datasets/val/labels")
            existing_instances = Classes.query.all()
            instance_names = [instance.name for instance in existing_instances]
            names_string = ', '.join(instance_names)
            training = TrainingData(epochs=t_epochs, batch_size=t_batch, patience=form.patience.data,
                                    learning_rate=t_lrf, precision=max_value, classes_detected=names_string)
            db.session.add(training)
            db.session.commit()
            subfolder_path = training.folder_name
            os.makedirs(subfolder_path)
            file_names = ['P_curve.png', 'PR_curve.png',
                          'R_curve.png', 'results.png', 'results.csv']
            copy_files_with_names(source_folder, subfolder_path, file_names)
            source_folder2 = 'runs/detect/train/weights'
            file_name = 'best.pt'
            copy_file_with_name(source_folder2, subfolder_path, file_name)
            print("Training.")
            print("Training..")
            print("Training...")
            form.epochs.data = ''
            form.batch_size.data = ''
            form.patience.data = ''
            form.learning_rate.data = ''
            flash("Trained Successfully!")
            return jsonify({"message": "Training started successfully!"}), 200
        else:
            # Log form errors
            print("Form validation failed:", form.errors)
            return jsonify({"error": "Form validation failed", "details": form.errors}), 400
    elif request.method == 'GET':
        Admin = "Admin"
        if Admin == "Admin":
            return render_template('trainForm.html', form=form)
        else:
            flash("Sorry, you must be an Admin to access this page")
            return redirect(url_for('dashboard'))
    else:
        return jsonify({"error": "Invalid request method"}), 405


####### Notifications Route #######


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
    source_image_file = image_to_add.frame.replace("images", "raw_images")
    destination_image_file = source_image_file.replace(
        "notifications/raw_images/", "updateDatasetFile/image/")
    shutil.copy2(source_image_file, destination_image_file)
    if os.path.exists(source_image_file):
        os.remove(source_image_file)
    print(source_image_file)
    raw_notif_annotation = source_image_file.replace(
        "raw_images", "raw_annotations")
    raw_notif_annotation = raw_notif_annotation.replace(".jpg", ".txt")
    if os.path.exists(raw_notif_annotation):
        os.remove(raw_notif_annotation)
    print(raw_notif_annotation)
    if os.path.exists(image_to_add.frame):
        os.remove(image_to_add.frame)
    print(image_to_add.frame)
    upload_folder = "static/DataSetimages"
    Dataset_image_folder = "datasets/all/images/"
    image_folder = "static/updateDatasetFile/image/"
    copy_files_in_folder(image_folder, Dataset_image_folder)
    copy_files_in_folder(image_folder, upload_folder)
    delete_files_in_folder(image_folder)
    db.session.delete(image_to_add)
    db.session.commit()
    flash(f'{image_to_add.timestamp} has been added.', 'success')
    return redirect(url_for('notifications'))

####### notifications 2 #######


@app.route('/notifications/<int:notification_id>/addtoDataset_notification', methods=['POST'])
@login_required
def addtoDataset_notification(notification_id):
    image_to_add = Notifications.query.filter_by(id=notification_id).first()
    source_image_file = image_to_add.frame.replace("images", "raw_images")
    source_image_annotation = source_image_file.replace(
        "raw_images", "raw_annotations")
    source_image_annotation = source_image_annotation.replace(".jpg", ".txt")
    destination_image_file = source_image_file.replace(
        "notifications/raw_images/", "updateDatasetFile/image/")
    destination_annotation_file = source_image_annotation.replace(
        "notifications/raw_annotations/", "updateDatasetFile/label/")
    shutil.copy2(source_image_file, destination_image_file)
    shutil.copy2(source_image_annotation, destination_annotation_file)
    if os.path.exists(source_image_file):
        os.remove(source_image_file)
    print(source_image_file)
    raw_notif_annotation = source_image_file.replace(
        "raw_images", "raw_annotations")
    raw_notif_annotation = raw_notif_annotation.replace(".jpg", ".txt")
    if os.path.exists(raw_notif_annotation):
        os.remove(raw_notif_annotation)
    print(raw_notif_annotation)
    if os.path.exists(image_to_add.frame):
        os.remove(image_to_add.frame)
    print(image_to_add.frame)
    upload_folder = "static/DataSetimages"
    Dataset_image_folder = "datasets/all/images/"
    Dataset_label_folder = "datasets/all/labels/"
    image_folder = "static/updateDatasetFile/image/"
    label_folder = "static/updateDatasetFile/label/"
    class_objects = Classes.query.all()
    class_names = [obj.name for obj in class_objects]
    process_images(image_folder, label_folder, upload_folder, class_names)
    copy_files_in_folder(image_folder, Dataset_image_folder)
    copy_files_in_folder(label_folder, Dataset_label_folder)
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
    notification_to_delete = Notifications.query.filter_by(
        id=notification_id).first()
    raw_notif = notification_to_delete.frame.replace("images", "raw_images")
    if os.path.exists(raw_notif):
        os.remove(raw_notif)
    raw_notif_annotation = raw_notif.replace("raw_images", "raw_annotations")
    raw_notif_annotation = raw_notif_annotation.replace(".jpg", ".txt")
    if os.path.exists(raw_notif_annotation):
        os.remove(raw_notif_annotation)
    if os.path.exists(notification_to_delete.frame):
        os.remove(notification_to_delete.frame)
    db.session.delete(notification_to_delete)
    db.session.commit()
    flash(f'{notification_to_delete.timestamp} has been deleted.', 'success')
    return redirect(url_for('notifications'))

####### notifications 4 #######


@app.route('/notifications/<int:notification_id>/download_notif', methods=['GET', 'POST'])
@login_required
def download_notif(notification_id):
    if request.method == 'POST' or 'GET':
        delete_files_in_folder('uploads/')
        image_to_add = Notifications.query.filter_by(
            id=notification_id).first()
        source_image_file = image_to_add.frame.replace("images", "raw_images")
        print(source_image_file)
        image_path, image_name = os.path.split(source_image_file)
        uploads_folder = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'uploads')
        destination_image_file = source_image_file.replace(
            "static/notifications/raw_images/", "uploads/")
        print(destination_image_file)
        shutil.copy2(source_image_file, destination_image_file)
        response = make_response(send_from_directory(
            directory=uploads_folder, path=image_name, as_attachment=True))
        response.headers['Content-Disposition'] = f'attachment; filename={image_name}'
        raw_notif = image_to_add.frame.replace("images", "raw_images")
        if os.path.exists(raw_notif):
            os.remove(raw_notif)
        raw_notif_annotation = raw_notif.replace(
            "raw_images", "raw_annotations")
        raw_notif_annotation = raw_notif_annotation.replace(".jpg", ".txt")
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


@app.route('/uploadmulti', methods=['POST'])
@login_required
def uploadmulti():
    if 'images' not in request.files or 'annotations' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    images = request.files.getlist('images')
    annotations = request.files.getlist('annotations')

    image_folder = "static/updateDataset/images/"
    label_folder = "static/updateDataset/labels/"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for image in images:
        if image and allowed_image(image.filename):
            imagename = secure_filename(image.filename)
            image.save(os.path.join(image_folder, imagename))

    for annotation in annotations:
        if annotation and annotation.filename.endswith('.txt'):
            filename = secure_filename(annotation.filename)
            annotation.save(os.path.join(label_folder, filename))

    upload_folder = "static/DataSetimages"
    Dataset_image_folder = "datasets/all/images/"
    Dataset_label_folder = "datasets/all/labels/"
    class_objects = Classes.query.all()
    class_names = [obj.name for obj in class_objects]

    process_images(image_folder, label_folder, upload_folder, class_names)
    copy_files_in_folder(image_folder, Dataset_image_folder)
    copy_files_in_folder(label_folder, Dataset_label_folder)
    delete_files_in_folder(image_folder)
    delete_files_in_folder(label_folder)

    return jsonify({'message': 'Files uploaded successfully!'}), 200


####### Dataset Route #######


@app.route('/dataset', methods=['GET'])
def dataset():
    img_folder = "static/DataSetimages/"
    images = os.listdir(img_folder)
    image_urls = [os.path.join(img_folder, image) for image in images]
    return jsonify(image_urls=image_urls)


@app.route('/delete_dataset_images', methods=['POST'])
def delete_dataset_images():
    delete_files_in_folder('static/DataSetimages/')
    delete_files_in_folder('datasets/all/images/')
    delete_files_in_folder('datasets/all/labels/')
    return jsonify({'message': 'Dataset successfully deleted'}), 200


def process_images(image_folder, label_folder, output_folder, className):
    image_files = [f for f in os.listdir(
        image_folder) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    os.makedirs(output_folder, exist_ok=True)
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        if label_file in label_files:
            label_file_path = os.path.join(label_folder, label_file)
            annotations = extract_variables(label_file_path)
            image_file_path = os.path.join(image_folder, image_file)
            image = PIL.Image.open(image_file_path)
            new_image = transform_to_annotated_img(
                image, annotations, className)
            new_image_file_path = os.path.join(output_folder, image_file)
            cv2.imwrite(new_image_file_path, new_image)


def process_images_withoutlabel(image_folder, output_folder):
    image_files = [f for f in os.listdir(
        image_folder) if f.endswith(('.jpg', '.png'))]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_file in image_files:
        image_file_path = os.path.join(image_folder, image_file)
        image = PIL.Image.open(image_file_path)
        new_image_file_path = os.path.join(output_folder, image_file)
        cv2.imwrite(new_image_file_path, image)


def delete_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def copy_files_in_folder(source_folder, destination_folder):
    file_list = os.listdir(source_folder)
    for file_name in file_list:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copy2(source_file, destination_file)


def copy_file_in_folder(source_folder, destination_folder, file_name):
    source_file = os.path.join(source_folder, file_name)
    destination_file = os.path.join(destination_folder, file_name)
    shutil.copy2(source_file, destination_file)


def extract_variables(file_path):
    variables = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 5:
                variables.append(parts)
    return variables


def transform_to_annotated_img(image, annotation, className):
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
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(new_image, (x_min, y_min),
                      (x_max, y_max), (255, 0, 255), 3)
        cv2.putText(new_image, className[int(
            class_label)], (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 3)
    return new_image


def video_detect(img, classNames):
    # Fetch the current confidence value
    setting = GeneralSettings.query.first()
    if setting:
        current_confidence = setting.confidence
    else:
        # Default to 0.5 if not found
        current_confidence = 0.5

    model = YOLO("models/model_used/best.pt")
    oil_leak_detected = False
    oil_leak_count = 0
    results = model(img, stream=True)
    image = PIL.Image.fromarray(img)
    img_width, img_height = image.size
    annotation = [0, 0, 0, 0, 0]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > current_confidence:  # Use the dynamic confidence value
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1,
                            [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                annotation = [cls, x_center, y_center, width, height]
                oil_leak_detected = True

    return img, oil_leak_detected, annotation


def video_detection(path_x):
    with app.app_context():
        video_capture = path_x
        cap = cv2.VideoCapture(video_capture)
        frame_width = int(cap.get(3))
        frame_width = int(cap.get(4))
        while True:
            success, img = cap.read()
            if not success:
                break
            image_clear = img.copy()
            class_objects = Classes.query.all()
            classNames = [obj.name for obj in class_objects]
            img, oil_leak_detected, annotation = video_detect(img, classNames)
            print(oil_leak_detected)
            if oil_leak_detected:
                current_time = datetime.now().strftime(
                    "%Y-%m-%d__%H_%M_%S_%f")[:-1]
                relative_path = "static/notifications/images/"
                img_name = relative_path + str(current_time) + ".jpg"
                msg = "Detection on Camera 1"
                notif = Notifications.query.order_by(
                    Notifications.id.desc()).first()
                notifBool = (notif.timestamp >= (datetime.now() -
                             timedelta(minutes=1))) if notif else False
                if not notifBool:
                    print("current time is " + current_time)
                    print(msg)
                    print(img_name)
                    img_name2 = img_name.replace("images", "raw_images")
                    file_name = img_name2.replace(
                        "raw_images", "raw_annotations")
                    file_name = file_name.replace(".jpg", ".txt")
                    print(len(annotation))
                    print("annotation cls: " + str(annotation[0]) + "x_center " + str(
                        annotation[1]) + "y_center " + str(annotation[2]) + "width " + str(annotation[3]) + "height ")
                    with open(file_name, "w") as file:
                        for num in annotation:
                            float_str = str(num)
                            file.write(float_str)
                            file.write(" ")
                    file.close()
                    img_name2 = img_name.replace("images", "raw_images")
                    cv2.imwrite(img_name2, image_clear)
                    cv2.imwrite(img_name, img)
                    notification = Notifications(message=msg, frame=img_name)
                    db.session.add(notification)
                    db.session.commit()
            yield img


def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

####### Models Route #######


@app.route('/models', methods=['GET'])
# @login_required
def models():
    model_list = TrainingData.query.all()
    return jsonify([model.to_dict() for model in model_list]), 200


####### Settings Route #######
class GeneralSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confidence = db.Column(db.Float, nullable=False, default=0.5)

    def __repr__(self):
        return f'<GeneralSettings confidence={self.confidence}>'


@app.route('/settings/confidence', methods=['GET', 'POST'])
def confidence_setting():
    if request.method == 'POST':
        new_confidence = request.json.get('confidence')
        if new_confidence is not None and 0 <= new_confidence <= 1:
            setting = GeneralSettings.query.first()
            if not setting:
                setting = GeneralSettings(confidence=new_confidence)
                db.session.add(setting)
            else:
                setting.confidence = new_confidence
            db.session.commit()
            return jsonify({'message': 'Confidence updated successfully', 'confidence': setting.confidence}), 200
        else:
            return jsonify({'error': 'Invalid confidence value'}), 400
    else:
        setting = GeneralSettings.query.first()
        if setting:
            return jsonify({'confidence': setting.confidence}), 200
        else:
            return jsonify({'error': 'Confidence setting not found'}), 404


def reset_selected():
    all_data = TrainingData.query.all()
    for data in all_data:
        data.selected = "False"
    db.session.commit()


@app.route('/settings', methods=['GET', 'POST'])
# @login_required
def settings():
    # Admin = current_user.role
    if request.method == 'POST':
        new_model_folder = request.json.get('model')
        if not new_model_folder:

            return jsonify({"error": "No model folder specified"}), 400

        selected_model = TrainingData.query.filter_by(
            folder_name=new_model_folder).first()
        if not selected_model:
            return jsonify({"error": "Model not found"}), 404

        try:
            reset_selected()
            selected_model.selected = "True"
            db.session.commit()
            detection_model_folder = "models/model_used/"
            selected_model_folder = selected_model.folder_name
            delete_files_in_folder(detection_model_folder)
            copy_file_in_folder(selected_model_folder,
                                detection_model_folder, "best.pt")
            class_list = Classes.query.all()
            return jsonify({"message": "Settings saved successfully!"}), 200
        except Exception as e:

            return jsonify({"error": str(e)}), 500
    else:
        model_list = TrainingData.query.all()
        # if Admin == "Admin":
        return jsonify([model.to_dict() for model in model_list]), 200

        # return render_template('settings.html', models=model_list)
        # else:
        # flash("Sorry, you must be an Admin to access this page")
        # return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)
