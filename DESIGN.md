# Data Adapter

## Purpose
The adapter serves as a way to gain real order book snapshot data from the Coinbase Level2 Channel and translate that data into a form that is useable by the book builder. We must build an adapter that handles incoming message sequences and prepares that data to be useable for building books, which requires identical inputs regardless of where the data comes from.

## Message Flow
Messages over WebSocket follow the following order: on the user end, a connection is made, then a subscription is sent. Next, Coinbase sends a snapshot that can be used to initialize the order book. Then Coinbase sends update messages containing information that can be used to update the data. The adapter takes this data and translates it into the form expected by the BookUpdate class. Once BookUpdate objects are assembled, they are passed into the BookBuilder class (which requires a consistent input type).

## Internal Representation
The BookUpdate class is setup using a NamedTuple, which is immutable to prevent accidental updates. The attributes of this class are the following:
```
side (str) - whether the update is on the bid or ask side of the book
price (float) - the price level this update applies to
size (float) - the current quantity available at the price level; size 0 indicates the level should be removed
time (Optional[str]) - the current time of the order book snapshot, recorded by Coinbase
```

## Message Routing
The adapter will handle messages including snapshots and l2updates. Messages that are unrecognized are ignored. We don't want to raise an error in this case because we don't want to halt the program; we want it to continue receiving the messages it is designed for. 

## Technology Choices
I am using the websockets library because it is built around the Asynchronous paradigm so that while a task waits for an external input, the rest of the code can continue running. This idea is powered by the asyncio library. This is important when waiting for update messages from Coinbase because we don't want to stall the rest of the code while waiting for a response. 
