from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv,find_dotenv
import os
from dotenv import load_dotenv,find_dotenv


app = Flask(__name__)
app.secret_key = '123456789'  
CORS(app)


mongo_url=os.getenv("mongo_url")
client = MongoClient(mongo_url)
db = client.flask_db
db.users.insert_one({"dummy": "data"})
db.users.delete_one({"dummy": "data"})
db.likes.insert_one({"dummy": "data"})
db.likes.delete_one({"dummy": "data"})
db.playlists.insert_one({"dummy": "data"})
db.playlists.delete_one({"dummy": "data"})
db.playlists_name.insert_one({"dummy": "data"})
db.playlists_name.delete_one({"dummy": "data"})
users = db.users
likes = db.likes
playlists = db.playlists
playlists_name=db.playlists_name




@app.route('/', methods=('GET', 'POST'))
def login():
    msg="Login"
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        myquery = { "email": email }
        mydoc = users.find_one(myquery)
   
        if mydoc:
            if password==mydoc['password']:
                session['email']=email
                return redirect(url_for('dashboard'))
            else:
                msg="Incorrect Password"
        else:
            msg="Email does not exist "


    return render_template('login.html',msg=msg)


@app.route('/signup', methods=('GET', 'POST'))
def signup():
    msg="Signup"
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']
        myquery = { "email": email }
        mydoc = likes.find_one(myquery)
        
        if mydoc:
            msg="Email Already Exist"
            
        else:
            msg="Signup Success"
            mydict = { "username": username, "email": email,"password":password }
            likes.insert_one(mydict)

            return redirect(url_for('login'))


 
    return render_template('signup.html',msg=msg)

@app.route('/dashboard', methods=('GET', 'POST'))
def dashboard():
  
    return render_template('dashboard.html')

@app.route('/like_update', methods=['POST'])
def like_update():
    data = request.get_json()
    url=data['url']
    email=session['email']
    cursor=likes.find_one({'email':email,'url':url})
    if cursor!=None:
        if cursor['count']==1:
            count=0
        else:
            count=1
        myquery = {'email':email,'url':url}
        newvalues = { "$set": { "count": count } }   
        likes.update_one(myquery, newvalues)
    else:
        mydict = { 'email':email, 'url': url,'count':1 }
        likes.insert_one(mydict)
    
    return jsonify({'message': 'Data received successfully'})

@app.route('/get_count', methods=['POST'])
def get_count():
    data = request.get_json()
    email=session['email']
    url=data['url']
    cursor=likes.find({'url':url})
    cursor=list(cursor)
    count=0
    color='grey'
    if cursor:
        for doc in cursor:
            if doc['count']==1:
                count+=1
                if doc['email']==email:
                    color='red'
    
       
       
    return jsonify({'count': count,'color':color})
    


@app.route('/logout')
def logout():
    session.pop('email', None)  
    session.clear() 
    return redirect(url_for('login'))


@app.route('/playlist')
def playlist():
    
    return render_template('playlist.html')


@app.route('/create_playlist', methods=['POST'])
def create_playlist():
    data = request.get_json()
    email=session['email']
    name = data['c_playlist']
    img=data['upload_image']
    mydict = { "name": name, "email": email,"image":img }
    playlists_name.insert_one(mydict)
  
    return jsonify({'message':'got it'})


@app.route('/get_names', methods=['POST'])
def get_names():
    data = request.get_json()
    email=session['email']
    query={'email':email}
    names_cursor = playlists_name.distinct('name',query)
    names = list(names_cursor)

   
    for name in names:
        if '_id' in name:
            name['_id'] = str(name['_id'])

  
    return jsonify({'names':names})

@app.route('/add_song', methods=['POST'])
def add_song():
    email=session['email']
  
 
    selected_playlists =  request.get_json()
   
    url=selected_playlists['url']
    img_url=selected_playlists['img']
    songname=selected_playlists['name']
    



    for name in selected_playlists:
        if name!='url':
            if name!='img':
                myquery = {'email':email,'url':url,'image':img_url,'playlist_name':name,'songname':songname}
                playlists.insert_one(myquery)
                
   

    clean()
    return jsonify({'message':'got it'})



@app.route('/get_playlists', methods=['POST'])
def get_playlists():
 
    email=session['email']
    query={'email':email}
    names_cursor = playlists_name.find(query)
    names = list(names_cursor)
    duplicate=[]
    filtered=[]

    for name in names:
        if '_id' in name:
            name['_id'] = str(name['_id'])
        if name['name'] not in duplicate:
            duplicate.append(name['name'])
            filtered.append(name)
            
    
  
    return jsonify({'names':filtered})

@app.route('/get_songs', methods=['POST'])
def get_songs():
    data = request.get_json()
   
  
    session['songname']=data['songname']


    return jsonify({'message':'got it'})

    
@app.route('/play_songs', methods=('GET', 'POST'))
def play_songs():
    return render_template('playlist_songs.html')



@app.route('/songs', methods=['POST'])
def songs():
    email=session['email']
    name=session['songname']
    image_cursor=playlists_name.find_one({'email':email,'name':name})
    img=image_cursor['image']
    query={'email':email,'playlist_name':name}
    cursor=playlists.find(query)
    cursor=list(cursor)
    duplicate=[]
    filtered=[]
    for doc in cursor:
        if '_id' in doc:
            doc['_id'] = str(doc['_id'])
        if doc['songname'] not in duplicate:
            duplicate.append(doc['songname'])
            filtered.append(doc)
    
    return jsonify({'cursor':filtered,'name':name,'image':img})


def clean():
    session.pop('imgurl', None) 
    session.pop('songname', None)  

    
    return 

if __name__ == '__main__':
    app.run(debug=True)
