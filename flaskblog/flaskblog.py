from flask import Flask, escape, request,render_template,url_for,flash,redirect
from forms import RegistrationForm, LoginForm
app = Flask(__name__)
app.config['SECRET_KEY']='3f7248fec3176977ff519a4650be47d4'
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
