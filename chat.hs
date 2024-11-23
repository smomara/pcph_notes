{-# LANGUAGE RecordWildCards #-}

import Network.Socket hiding (Broadcast)
import Control.Monad
import Control.Concurrent
import System.IO
import Text.Printf
import Control.Exception hiding (handle)
import Control.Concurrent.Async
import Control.Concurrent.STM
import qualified Data.Map as Map
import Data.Map (Map)

---------------------
-- | CLIENT DATA | --
---------------------
type ClientName = String

data Client = Client
  { clientName :: ClientName -- username
  , clientHandle :: Handle -- client's buffer
  , clientKicked :: TVar (Maybe String) -- Nothing if not kicked
                                        -- Just [reason kicked] if kicked
  , clientSendChan :: TChan Message -- carries messages sent to client
  , clientBroadcastChan :: TChan Message -- broadcast chan copy
  }

data Message = Notice String -- message from the server
             | Tell ClientName String -- private message
             | Broadcast ClientName String -- public message from another client
             | Command String -- line of text received from the user

-- | Construct a new instance of Client
newClient :: ClientName -> Handle -> Server -> STM Client
newClient name handle Server{..} = do
  c <- newTChan
  k <- newTVar Nothing
  bc <- dupTChan broadcastChan
  return Client
    { clientName = name
    , clientHandle = handle
    , clientKicked = k
    , clientSendChan = c
    , clientBroadcastChan = bc
    }

-- | Sends a Message to a given Client
sendMessage :: Client -> Message -> STM ()
sendMessage Client{..} msg = writeTChan clientSendChan msg

---------------------
-- | SERVER DATA | --
---------------------

-- data that must be accessible to all clients
data Server = Server
  { clients :: TVar (Map ClientName Client)
  , broadcastChan :: TChan Message
  }

-- | Construct a new instance of Server
newServer :: IO Server
newServer = atomically $ do
  c <- newTVar Map.empty
  bc <- newTChan
  return Server 
    { clients = c 
    , broadcastChan = bc
    }

-- | broadcast a Message to the broadcast channel
broadcast :: Server -> Message -> STM ()
broadcast Server{..} msg = writeTChan broadcastChan msg

-- | kick a client from the server
-- TODO: add reason
kick :: Server -> ClientName -> ClientName -> STM ()
kick server@Server{..} who by = do
  clientMap <- readTVar clients
  case Map.lookup who clientMap of
    Nothing ->
      void $ sendMessage (clientMap Map.! by) $
        Notice (who ++ " is not connected")
    Just victim -> do
      writeTVar (clientKicked victim) $ Just ("by " ++ by)
      broadcast server $ Notice (who ++ " was kicked by " ++ by)

-- | send a private message
tell :: Server -> Client -> ClientName -> String -> IO ()
tell Server{..} from who msg = atomically $ do
  clientMap <- readTVar clients
  case Map.lookup who clientMap of
    Nothing -> sendMessage (clientMap Map.! (clientName from)) $
      Notice (who ++ " is not connected")
    Just to -> do 
      sendMessage to $ Tell (clientName from) msg
      sendMessage from $ Tell (clientName from) msg

--------------------
-- | THE SERVER | --
--------------------

port :: Int
port = 44444

resolve :: IO SockAddr
resolve = do
  let hints = defaultHints {
        addrFlags = [AI_PASSIVE],
        addrSocketType = Stream
      }
  addr:_ <- getAddrInfo (Just hints) Nothing (Just $ show port)
  return $ addrAddress addr

main :: IO ()
main = withSocketsDo $ do
  server <- newServer
  addr <- resolve
  sock <- socket AF_INET Stream defaultProtocol
  bind sock addr
  listen sock 5
  printf "Listening on port %d\n" port
  forever $ do
    (conn, sockAddr) <- accept sock
    handle <- socketToHandle conn ReadWriteMode
    printf "Accepted connection from %s\n" (show sockAddr)
    forkFinally (talk handle server) (\_ -> hClose handle)

---------------------------------
-- | SETTING UP A NEW CLIENT | --
---------------------------------

-- | attempts to add a new client with given name to the state
checkAddClient :: Server -> ClientName -> Handle -> IO (Maybe Client)
checkAddClient server@Server{..} name handle = atomically $ do
  clientMap <- readTVar clients
  if Map.member name clientMap
     then return Nothing
     else do client <- newClient name handle server
             writeTVar clients $ Map.insert name client clientMap
             broadcast server $ Notice (name ++ " has connected")
             return (Just client)

-- | removes client corresponding to given name
removeClient :: Server -> ClientName -> IO ()
removeClient server@Server{..} name = atomically $ do
  modifyTVar clients $ Map.delete name
  broadcast server $ Notice (name ++ " has disconnected")

-- (rolling our own logic instead of bracket, since checkAddClient is conditional)
-- (ref: bracket definition)
talk :: Handle -> Server -> IO ()
talk handle server = do
  hSetNewlineMode handle universalNewlineMode
  hSetBuffering handle LineBuffering
  readName
 where
  readName = do
    hPutStrLn handle "What is your name?"
    name <- hGetLine handle
    if null name
      then readName
      else mask $ \_ -> do
          ok <- checkAddClient server name handle
          case ok of
            Nothing -> do
              hPrintf handle "The name %s is in use, please choose another\n" name
              readName
            Just client ->
              runClient server client
                `finally` removeClient server name

----------------------------
-- | RUNNING THE CLIENT | --
----------------------------

-- | main client functionality
runClient :: Server -> Client -> IO ()
runClient serv client@Client{..} = do
  _ <- race server receive
  return ()
 where
  receive = forever $ do
    msg <- hGetLine clientHandle
    atomically $ sendMessage client (Command msg)

  server = join $ atomically $ do
    k <- readTVar clientKicked
    case k of
      Just reason -> return $ hPutStrLn clientHandle ("You have been kicked: " ++ reason)
      Nothing -> do
        msg <- readTChan clientBroadcastChan `orElse` readTChan clientSendChan
        return $ do
          continue <- handleMessage serv client msg
          when continue $ server

-- | acts on a message
handleMessage :: Server -> Client -> Message -> IO Bool
handleMessage server client@Client{..} message =
  case message of
    Notice msg -> output $ "*** " ++ msg
    Tell name msg -> output $ "*" ++ name ++ "*: " ++ msg
    Broadcast name msg -> output $ "<" ++ name ++ ">: " ++ msg
    Command msg ->
      case words msg of
        ["/kick", who] -> do
          atomically $ kick server who clientName
          return True
        "/tell" : who : what -> do
          tell server client who (unwords what)
          return True
        ["/quit"] ->
          return False
        ('/':_):_ -> do
          hPutStrLn clientHandle $ "Unrecognized command: " ++ msg
          return True
        _ -> do
          atomically $ broadcast server $ Broadcast clientName msg
          return True
 where
  output s = do hPutStrLn clientHandle s; return True
