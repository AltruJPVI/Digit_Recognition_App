- This is a hand written digit recognition app.
- The main framenwork I used to create the CNN is PyTorch.
- For the interactive app I used streamlit because its easy and i don't really want to learn frontend.
- There is a model I trained which works really good.
- The dataset itself is very complete and clean, a pretty small CNN performs really good.

The difficult part is to make it work in a real scenario were you have to recognise new digits on live. Why? because the datasets has perfectly centered, equaliy
zoomed and well written. This makes it difficult for the model to generalize in a real situation were the digits written migth be big,small, in a corner, a bit rotated....
So in order to have realistic results we want to make image augmentation to recreate a real situation.
