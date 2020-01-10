from flask import Flask, escape, request,render_template,url_for,flash,redirect
from forms import RegistrationForm, LoginForm
from flask_sqlalchemy import SQLAlchemy
import datetime

app = Flask(__name__)
app.config['SECRET_KEY']='3f7248fec3176977ff519a4650be47d4'
# Setting up the URI and sqlite as the database
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    username=db.Column(db.String(20),unique=True,nullable=False)
    email=db.Column(db.String(120),unique=True,nullable=False)
    image_file=db.Column(db.String(20),nullable=False,default='default.jpg')
    password=db.Column(db.String(60),nullable=False)
    posts=db.relationship('Post',backref='author',lazy=True)
    def __repr__(self):
        return f"User('{self.username}','{self.email}','{self.image_file}')"

class Post(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    title=db.Column(db.String(100),nullable=False)
    date_posted=db.Column(db.DateTime,nullable=False,default=datetime.utcnow())
    content=db.Column(db.Text,unique=True,nullable=False)
    def __repr__(self):
        return f"Post('{self.title}','{self.date_posted}','{self.image_file}')"


posts=[
    {
    'author':'CS',
    'title':'Blog Post 1',
    'content':'First post content',
    'date_posted':'April 9, 2019'
    },
    {
    'author':'John',
    'title':'Blog Post 2',
    'content':'Second post content',
    'date_posted':'October 9, 2019'
    }

]

@app.route('/')
@app.route('/homepage')
def home():
    name = request.args.get("name", "World")
    return render_template('home.html',posts=posts)
@app.route('/about')
def about():
    return render_template('about.html',title="About")

@app.route('/register',methods=['POST','GET'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}",'success')
        return redirect(url_for('home'))
    return render_template('register.html',title="Register",form=form)

@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == "admin@blog.com" and form.password.data == "password":
            flash(f"You Have logged in!",'success')
            return redirect(url_for('home'))
        else:
            flash(f"Login Unsuccessful. Please check username and password",'danger')
    return render_template('login.html',title="Login",form=form)


if __name__=='__main__':
    app.run(debug=True)
