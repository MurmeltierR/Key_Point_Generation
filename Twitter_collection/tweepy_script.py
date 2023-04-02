import requests
import os
import json
import time
from datetime import datetime, timedelta 
import pandas as pd
import tweepy
from pandas.io.json import json_normalize
from sqlalchemy import create_engine
import pymysql

#Twitter API Credentials
API_KEY = ''
SECRET_KEY = ''
BEARER_TOKEN = ''
ACCESS_TOKEN = ''
SECRET_TOKEN = ''

#establish connecion to Twitter API
auth = tweepy.OAuthHandler(API_KEY, SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, SECRET_TOKEN)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

#Connect to Database sqlalchemy + pymysql
sqlEngine       = create_engine('mysql+pymysql://username:password@127.0.0.1/database')
dbConnection    = sqlEngine.connect()
connection      = pymysql.connect(host='localhost',
                         user='username',
                         password='password',
                         db='database')
cursor=connection.cursor()


#read the mp file and initiate dataframe 
mp_df = pd.read_csv('Path/to/data/MPsonTwitter.csv')
mp_df = mp_df['Screen name'].str.replace('@', '')
mp_list = mp_df.values.tolist()
tweets_df = pd.DataFrame([])

#method to read resultset of Tweepy
def jsonify_tweepy(tweepy_object):
    json_str = json.dumps(tweepy_object._json)
    return json.loads(json_str)


#for each MP in list lookup latest tweets in database and get latest tweets
for mp in mp_list:
    sql = "SELECT distinct(id) FROM uk_mps WHERE user_screen_name = " + "'" + mp[1] + "'" + " AND created_at = (SELECT MAX(created_at) FROM uk_mps WHERE user_screen_name =  " + "'" + mp[1] + "'" + " )"
    cursor.execute(sql)
    result = cursor.fetchall()
    try:
        recent_tweet = api.user_timeline(screen_name = mp[1], since_id=result[0][0], tweet_mode = 'extended')
        tweet = [jsonify_tweepy(follower) for follower in recent_tweet]
        tweets_df = tweets_df.append(pd.json_normalize(tweet))
    except tweepy.TweepyException:
        pass

#clear resulting dataframe
tweets_df.drop(tweets_df.columns.difference(['created_at','id', 'id_str','text','user.id','user.name','user.screen_name', 'user.followers_count']), 1, inplace=True)
cleaned_df = tweets_df.rename(columns={'user.id': 'user_id', 'user.name': 'user_name', 'user.screen_name':'user_screen_name', 'user.followers_count': 'user_followers_count'})
cleaned_df['created_at'] = cleaned_df['created_at'].map(lambda x: time.strptime(x,'%a %b %d %H:%M:%S +0000 %Y'))

#write resultset to database
cleaned_df.to_sql('uk_mps', dbConnection, index=False,if_exists='append')
dbConnection.commit()

#close connection
connection.close()
dbConnection.close()

now = datetime.now()
date_path = now.strftime("%Y-%m-%d")
base_dir = 'Path/to/data/'
outdir = os.path.join(base_dir, date_path, '')

if not os.path.exists(outdir):
    os.makedirs(outdir)

tweets_df.to_json(outdir +'%s.json' % (time.strftime('%H-%M')), orient='records')