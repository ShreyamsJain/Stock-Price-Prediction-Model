import praw

reddit = praw.Reddit(client_id='fp8nCSTNLNXtmA',
                     client_secret='idMdxXwmokXGa-Dv7kfhpiLcP58',
                     password='1Reddit1$',
                     user_agent='Comment Extraction (by /u/USERNAME)',
                     username='shreyams')
print(reddit.user.me())

count = 0
'''
subreddit = reddit.subreddit('worldnews')
for submission in subreddit.submissions(1483228800, 1514764799).top('day'):
    count += 1
    print("\n\n\n\n" + str(count))
    print(submission.title)
#    print(submission.date)

print(count)

reddit.redditor('spez').submissions.top('all')

'''
'''
for submission in reddit.subreddit('all').search('praw'):
    print(submission.title)
    '''
