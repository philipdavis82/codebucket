how to use:

open ipython in code directory

generate text from model

    [1]:import cookie_learner

    [2]:model = cookie_learner.learner()

    [3]:model.load_model()	#Needs to be called when generating text. only has to be called once.

    [4]:text = model.get_text("start string",(number between 0.1 - 2),(Number of characters))

    [5]:print(text)

train model from scratch

    [1]:import cookie_learner

    [2]:model = cookie_learner.learner()

    [3]:model.EPOCHS = 10 #number of itterations ~5min per itterations baised on computation power

    [4]:model.train_model()

train model from checkpoint

    [1]:import cookie_learner

    [2]:model = cookie_learner.learner()

    [3]:model.EPOCHS = 10 #number of itterations ~5min per itterations baised on computation power

    [4]:model.load_model_for_training()

    [5]:model.train_model()

train from new text

    >>replace cookie_data/full_set with the text file you want to train with

    [1]:import cookie_learner

    [2]:model = cookie_learner.learner()

    [3]:model.EPOCHS = 10 #number of itterations ~5min per itterations baised on computation power

    [4]:model.train_model()