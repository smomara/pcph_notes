# Parallel and Concurrent Programming in Haskell

## Introduction

### Parallel vs Concurrent

**Parallel**:
using a multiplicity of computational hardware (ex: several processor cores)
to perform a computation more quickly.

**Concurrent**:
using mulitple threads of control where each thread executes at the same time.

The difference is that parallel programming is concerned only with efficiency and speed,
whereas concurrent programming is concerned with structuring a program that needs to
interact with multiple external agents running the program at the same time.
For example, the user, the database, and some external clients.
All these agents will run the program at the same time,
but their threads will be distinct.

Threads of control does not make sense in pure functional,
since there are no side-effects - meaning the order of execution does not matter.
So concurrency is effective for code in the IO monad (effectful code)

### Deterministic vs Nondeterministic

**Deterministic**: each program can only give one result

**Nondeterministic**:
admits programs may have different results, depending on some aspect of the execution

Concurrent prgrams are necessarily nondeterministic because
they must interact with some external agents that cause events at unpredictable times.

Nondeterminism is significantly harder to test and reason about.

## Parallel Haskell

### Synchronization and Communication

Haskell programmer doesn't have to explicity deal with this.
The programmer indicates where the parallelism is, and the details of running the
program in parallel are left to the runtime system.

### Granularity and Data Dependencies

**Granularity**:
Granularity needs to be large enough to dwarf overhead, but not too large,
because then you risk no thaving enough work to keep all the processors busy.

**Data Dependencies**:
When one task depends on another, they must be performed sequentially.

### Parallel Programming Models in Haskell

1. `Eval` monad and Evaluation Strategies
    * suitable for expressing parallelism in programs that are not heavily numerical
      or array-based

2. `Par` monad
    * affords the programmer more control in exchange for some of the
      conciseness and modularity of Strategies

3. `Repa` library
    * provides a rich set of combinators for building parallel array computations
    * you can express a complex array algorithm as the composition of several
      simpler operatoins, and the library optimizes the composition into a
      single-pass algorithm using *fusion*

4. `Accelerate` on GPU
    * similar to `Repa` but on GPU

*Note: skipping `Repa` and `Accelerate` chapters in PCPH for now*

## Basic Parallelism: The Eval Monad

### Lazy Evaluation and Weak Head Normal Form

**Lazy**: expressions are not evaluated until they are required

In this example, `x` is unevaluated

```Haskell
ghci> let x = 1 + 2 :: Int
ghci> :sprint x
x = _
```

An object in memory representing an unevaluated computation is called a *thunk*.
Here, x is a pointer to an object in memory representing the function `+`
applied to the integers `1` and `2`.

Let's look at another example:

```Haskell
ghci> let x = 1 + 2 :: Int
ghci> import Data.Tuple
ghci> let z = swap (x,x+1)
ghci> :sprint z
z = _
ghci> seq z ()
()
ghci> :sprint z
z = (_,_)
```

In this example, we evaluate (with seq) the thunk that points to the swap function,
which evaluates `z` to a pair, but the components of the pair are still unevaluated.

The `seq` function evaluates its argument only as far as the first contructor,
and not any more than that. We call this evaluating to
*weak head normal form* (WHNF).
*Normal form* means fully evaluated.

### The Eval Monad, rpar, and rseq

We get basic parallel techniques from the module `Control.Parallel.Strategies`.
Let's look at some definitions:

```Haskell
data Eval a
instance Monad Eval

runEval :: Eval a -> a

rpar :: a -> Eval a
rseq :: a -> Eval a
```

Parallelism is expressed using the `Eval` monad, which comes with `rpar` and `rseq`.

`rpar` creates parallelism: its arguments can be evaluated in parallel.
Its arguments should be a thunk, so useful evaluation happens.

`rseq` forces sequential evaluation: evaluating arguments and waiting for results.

`runEval` performs the `Eval` computation and returns its result. It's pure.

Let's look at an example. Suppose `f x` takes longer to run than `f y`:

```Haskell
runEval $ do
    a <- rpar (f x)
    b <- rpar (f y)
    rseq a
    rseq b
    return (a,b)
```

This creates an Eval computation that sparks parallel evaluation of `f x` and `f y`,
then waits for both results sequentially before returning them.

## Evaluation Strategies

A `Strategy` is a function in the `Eval` monad that takes a value of
type `a` and returns the same value:

```Haskell
type Strategy a = a -> Eval a
```

The idea is that a `Strategy` takes a data structure as input,
traverses the structure creating parallelism with `rpar` and `rseq`,
and then returns the original value.

Here's a simple exapmle:

```Haskell
parPair :: Strategy (a, b)
parPair (a, b) = do
    a' <- rpar a
    b' <- rpar b
    return (a',b')

-- to evaluate:
runEval $ parPair (fib 35, fib 36)

-- or, more clean
using :: a -> Strategy a -> a
x `using` s = runEval (s x)

(fib 35, fib 36) `using` parPair
```

### Parameterized Strategies

A better way to factor things is to write a parameterized Strategy,
so we can do something different with a pair flexibly, insteading of writing
a whole new strategy.

Let's look at an example:

```Haskell
evalPair :: Strategy a -> Strategy b -> Strategy (a,b)
evalPair sa sb (a,b) = do
    a' <- sa a
    b' <- sb b
    return (a',b')

-- non-parameterized version of parPair that just uses rpar
parPair :: Strategy (a, b)
parPair = evalPair rpar rpar

rparWith :: Strategy a -> Strategy a

-- parameterized version of parPair
parPair :: Strategy a -> Strategy b -> Strategy (a,b)
parPair sa sb = evalPair (rparWith sa) (rparWith sb)
```

Now instead of only being to evaluate to WHNF, we can do anything we want flexibly.

### A Strategy for Evaluating a List in Parallel

`evalList` and `parList` are provided by `Control.Parallel.Strategies`,
but lets look at the implementations:

```Haskell
evalList :: Strategy a -> Strategy [a]
evalList strat [] = return []
evalList strat (x:xs) = do
    x' <- strat x
    xs' <- evalList strat xs
    return (x':xs')

parList :: Strategy a -> Strategy [a]
parList strat = evalList (rparWith strat)
```

`evalList` walks the list recursively, applying the `Strategy` parameter `strat`
to each of the elements and building the result list.

`parList` is a parameterized `Strategy` that takes as an argument
a `Strategy` on values of type `a` and returns a `Strategy` for lists of type `a`.
So, `parList` describes a family of Strategies on lists
that evaluate the list elements in parallel.

### Example: The K-Means Problem

We will use Lloyd's algorithm which first makes an initial guess
at the center of each cluster, then proceeds as follows:

1. Assign each point to the cluster of which its closest,
   yielding a new set of clusters
2. Find the *centroid* of each cluster (the average of all the points)
3. Repeast steps 1 and 2 until the cluster locations stabilize.
   We cut off porcessing after an arbitrary number of iterations,
   because sometimes the algorithm does not converge.

Here's a rundown of the core implementation:

A data point is represented by the type `Point`:

```Haskell
data Point = Point !Double !Double

zeroPoint :: Point
zeroPoint = Point 0 0

sqDistance :: Point -> Point -> Double
sqDistance (Point x1 y1) (Point x2 y2) = ((x1-x2)^2) + ((y1-y2^2))
```

A cluster is represented by the type `Cluster`:

```Haskell
data Cluster
    = Cluster { clId :: Int
              , clCent :: Point
              }
```

And we need an intermediate type called `PointSum` that represents a set of points;
it contains the number of points in the set and the sum of their x and y coords:

```Haskell
data PointSum = PointSum ~Int !Double !Double

-- its constructed incrementally with addToPointSum
addToPointSum :: PointSum -> Point -> PointSum
addToPointSum (Pointsum count xs ys) (Point x y)
    = PointSum (count+1) (xs+x) (ys+y)

-- can be turned into a Cluster by computing the centroid
pointSumToCluster :: Int -> PointSum -> Cluster
pointSumToCluster i (PointSum count xs ys) =
    Cluster { clId = i
            , clCent = Point (xs / fromIntegral count) (ys / fromIntegral count)
            }
```

In the program, the input is a set of points represented as `[Point]`,
and an initial guess represented as `[Cluster]`.
The algorithm will iteratively refine the clusters until convergence is reached.

The `assign` function implements step 1 of the algorithm,
assigning points to clusters and building a vector of `PointSums`:

```Haskell
assign :: Int -> [Cluster] -> [Point] -> Vector PointSum
assign nClusters clusters points = Vector.create $ do
    vec <- MVector.replicate nclusters (Pointsum 0 0 0)
    let
        addpoint p = do
            let c = nearest p; cid = clId c
            ps <- MVector.read vec cid
            MVector.write vec cid $! addToPointSum ps p

    mapM_ addpoint points
    return vec
  where
    nearest p = fst $ minimumBy (compare `on` snd)
                                [ (c, sqDistance (clCent c) p) | c <- clusters ]
```

The `makeNewClusters` function implements step 2 of the algorithm:

```Haskell
makeNewClusters :: Vector PointSum -> [Cluster]
makeNewClusters vec =
    [ pointSumToCluster i ps
    | (i,ps@(PointSum count _ _)) <- zip [0..] (Vector.toList vec)
    , count > 0
    ]
```

`step` combines `assign` and `makeNewClusters` to implement one complete iteration:

```Haskell
step :: Int -> [Cluster] -> [Point] -> [Cluster]
step nclusters clusters points
    = makeNewClusters (assign nclusters clusters points)
```

Here is the sequential implementation of a loop that repeatedly applies
the `step` function until convergence:

```Haskell
kmeans_seq :: Int -> [Point] -> [Cluster] -> IO [Cluster]
kmeans_seq nclusters points clusters =
    let
        loop :: Int -> [Cluster] -> IO [Cluster]
        loop n clusters | n > tooMany = do
            putStrLn "giving up"
            return clusters
        loop n clusters = do
            printf "iteration %d\n" n
            putStr (unlines (map show clusters))
            let clusters' = step nclusters clusters points
            if clusters' == clusters
                then return clusters
                else loop (n+1) clusters'
    in loop 0 clusters

tooMany = 80
```

#### Parallelizing K-Means

It looks profitable to parallelize `assign` because its essentialy a `map`.
The operations are very fine-grained, so a simple `parMap` or `parList`
would be too much overhead. We will split into chunks instead,
then process the chunks in parallel.

```Haskell
split :: Int -> [a] -> [[a]]
split numChunks xs = chunk (length xs `quot` numCHunks) xs

chunk :: Int -> [a] -> [[a]]
chunk n [] = []
chunk n xs = as : chunk n bs
    where (as,bs) = splitAt n xs
```

With the resulting list of `Vector PointSum`s that we will get after mapping
`assign` over each chunk, we need to combine into a single `Vector PointSum`:

```Haskell
addPointSums :: PointSum -> PointSum -> PointSum
addPointSums (PointSum c1 x1 y1) (PointSum c2 x2 y2)
    = PointSum (c1+c2) (x1+x2) (y1+y2)

combine :: Vector PointSum -> Vector PointSum -> Vector PointSum
combine = Vector.zipWith addPointSums
```

Now we have all the pieces to define a parallel version of `step` and the main loop:

```Haskell
parSteps_strat :: Int -> [Cluster] -> [[Point]] -> [Cluster]
parSteps_strat nclusters clusters pointss
    = makeNewClusters $
        foldr1 combine $
            (map (assign nclusters clusters) pointss
                `using` parList rseq)

kmeans_strat :: Int -> Int -> [Point] -> [Cluster] -> IO [Cluster]
kmeans_strat numChunks nclusters points clusters =
    let chunks = split numChunks points

        loop :: Int -> [Cluster] -> IO [Cluster]
        loop n clusters | n > tooMany = do
            printf "giving up"
            return clusters
        loop n clusters = do
            printf "iteration %d\n" n
            putStr (unlines (map show clusters))
            let clusters' = parSteps_strat nclusters clusters chunks
            if clusters' == clusters
                then return clusters
                else loop (n+1) clusters
    in loop 0 clusters
```

Now we can experiment with the number of chunks to find the best granularity.

### GC'd Sparks and Speculative Parallelism

**Speculative**: parallelism where results may or may not be needed

Runtime automatically discards unreferenced sparks,
avoiding wasted resources on unneeded computations.

Runtime provides spark statistics with `+RTS -s`

```shell
SPARKS: 1000 (2 converted, 0 overflowed, 0 dud, 998 GC'd, 0 fizzled)
```

High GC'd spark count usually indicates issues
(unless using intentional speculation)

#### Best Practices

* Always use `using` to apply Strategies instead of raw `runEval`

* All `Control.Parallel.Strategies` combinators handle spark references correctly,
    So use them when possible.

* When writing your own `Eval` monad code:

    ```Haskell
    -- Bad:
    do
        ...
        rpar (f x)

    -- Good:
    do
        ...
        y <- rpar (f x)
        ... y ...

    -- OK as long as y is required by the program somewhere
    do
        ...
        rpar y
        ...
    ```

### Parallelizing Lazy Streams with parBuffer

It's possible that parallelizing computations on a lazy stream will destroy
the lazy streaming property and the program will require space linear
in the size of the input.

Let's use RSA encryption and decryption as an example. The RSA program
is a stream transformer, consuming input and producing output lazily.

```Haskell
encrypt :: Integer -> Integer -> ByteString -> ByteString
encrypt n e = B.unlines
            . map (B.pack . show . power e n . code)
            . chunk (size n)
```

The `map` will be the target of our parallelization.
Let's try to use the `parList` Strategy:

```Haskell
encrypt n e = B.unlines
            . withStrategy (parList rdeepseq)
            . map (B.pack . show . power e n . code)
            . chunk (size n)
```

This gives us a decent speedup, but `parList` forces the whole spine of the list,
preventing the program from streaming in constant space.

However, `parBuffer` from `Control.Parallel.Strategies` solves this by taking
an `Int` argument as a buffer size, so it doesn't eagerly create sparks for
every element in the list like `parList`, but streams in the first *N* elements
and creates sparks for them, so theres always *N* sparks available:

```Haskell
parBuffer :: Int -> Strategy a -> Strategy [a]
```

Here's how we would use it for encrypt (50-500 is usually good for size):

```Haskell
encrypt n e = B.unlines
            . withStrategy (parBuffer 100 rdeepseq)
            . map (B.pack . show . power e n . code)
            . chunk (size n)
```

This gives us a huge speedup with a constant supply of sparks.

### Chunking Strategies

`Control.Parallel.Strategies` library provides a version of `parList`
that has chunking built in:

```Haskell
parListChunk :: Int -> Strategy a -> Strategy [a]
```

The first arg is the number of elements in each chunk.

This is useful when a list is too large to create a spark for each element,
or the list elements are too cheap to warrant a spark each.

When the spark pool is full, subsequent sparks are dropped and reported
as `overflowed` in the `+RTS -s` output. If there are overflowed sparks,
it is usually best to create fewer sparks with `parListChunk` instead of `parList`.

### The Identity Property

A Strategy must obey the identity property in order for

```Haskell
x `using` s
```

to be an equivalent parallelized version of the sequential `x`.
That is, the value it returns must be equal to the value it was passed.
This is enforced by convention only, so we must be aware when we write our own
`Eval` monad code.

## Dataflow Parallelism: The Par Monad

`Eval` monad and Strategies work in conjuction with lazy evaluation to
express parallelism. This allows the algorithm to be decoupled from the parallelism,
and allows parallel evaluation strategies to be built compositionally.
But we might not want to build a lazy data structure (it can be hard to
understand and diagnose performance when lazy).

The `Par` monad is more explicit about granularity and data dependencies,
and avoids the reliance on lazy evaluation without sacrificing determinism.

```Haskell
newtype Par a
instance Applicative par
intance Monad par

runPar :: Par a -> a

fork :: Par () -> Par ()
```

A computation in the `Par` monad can be run using `runPar` to produce a pure result.

`fork` is how we create parallel tasks.
The `Par` computation passed as the argument (the child) is executed in parallel
with the caller (the parent). But `fork` doesn't return anything to the parent,
so we need to use the `IVar` type and its operatoins to pass values between
`Par` computations.

```Haskell
data IVar a -- instance Eq

new :: Par (IVar a)
put :: NFData a => IVar a -> a -> Par ()
get :: IVar a -> Par a
```

An `IVar` is like a box that starts empty.
`put` stores a value in the box, and `get` reads the value.
If the `get` operation finds the box empty,
then it waits until the box is filled by a `put`.

So, you can put a value in the box in one place and get it in another.
Once filled, the box stays full. `get` doesn't remove the value.
You shouldn't `put` more than once on the same `IVar`.

An `IVar` is like a *promise*.

Let's look at an example:

```Haskell
runPar $ do
    i <- new
    j <- new
    fork (put i (fib n))
    fork (put j (fib m))
    a <- get i
    b <- get j
    return (a+b)
```

In this example, we wait for `a` and `b` to `get` the results
before evaluating `(a+b)`.

`fib n` and `fib m` are evaluating indepently, creating a *dataflow*,
this means they can be evaluated in parallel.

Here's another useful function from `Control.Monad.Par` that forks
a computation in parallel and returns an IVar that can be used to wait
for the result:

```Haskell
spawn :: NFData a => Par a -> Par (IVar a)
spawn p = do
    i <- new
    fork (do x <- p; put i x)
    return i
```

The library also provides a parallel map:

```Haskell
-- note that f returns its results in the Par monad,
-- so f itself can create further parallelism using fork and other Par operations
parMapM :: NFData b => (a -> Par b) -> [a] -> Par [b]
parMapM f as = do
    ibs <- mapM (spawn . f) as
    mapM get ibs

-- here's one where it takes a non-monadic f
parMap :: NFData b => (a -> b) -> [a] -> Par [b]
parMap f as = do
    ibs <- mapM (spawn . return . f) as
    mapM get ibs
```

### Example: Shortest Paths in a Graph

Floyd-Warshall algorithm finds the lengths of the shortest paths between all
pairs of nodes in a weighted directed graph.

Assuming vertices are numbered from one, and we have a function `weight g i j`
that gives the weight of the edge from `i` to `j` in graph `g`, the algorithm
is described by this pseudocode:

```Haskell
shortestPath :: Graph -> Vertex -> Vertex -> Vertex -> Weight
shortestPath g i j 0 = weight g i j
shortestPath g i j k = min (shortestPath g i j (k-1))
                           (shortestPath g i k (k-1)
                            + shortestPath g k j (k-1))
```

Think of `shortestPath g i j k` giving the length of the shortest path
from `i` to `j`, passing through vertices up to `k` only.

At `k == 0`, the paths between each pair of vertices consists of the direct edges only.
For a nonzero `k`, there are two cases:
either the shortes path from `i` to `j` passes through `k` or it does not.
The shortest path passing thorugh `k` is given by the sum of
the shortest path from `i` to `k` and from `k` to `j`.
Then, the shortes path from `i` to `j` is the minimum of the two choices,
either passing through `k` or not.

Here's the sequential implementation of the shortest path algorithm:

```Haskell
shortestPaths :: [Vertex] -> Graph -> Graph
shortestPaths vs g = foldl' update g vs
  where
    update g k = Map.mapWithKey shortmap g
      where
        shortmap :: Vertex -> IntMap Weight -> IntMap Weight
        shortmap i jmap = foldr shortest Map.empty vs
          where
            shortest j m =
                case (old,new) of
                    (Nothing, Nothing) -> m
                    (Nothing, Just w) -> Map.insert j w m
                    (Just w, Nothing) -> Map.insert j w m
                    (Just w1, Just w2) -> Map.insert j (min w1 w2) m
              where
                old = Map.lookup j jmap
                new = do w1 <- weight g i k
                         w2 <- weight g k j
                         return (w1+w2)
```

The algorithm is a nest of three loops.
The outer loop is a left-fold with a data dependency between iterations,
so it cannot be parallelized.
The next loop (`update`) is a map, which parallelizes nicely.

We want to behave like `parMap`, except we will have to use `traverseWithKey`
since we want to map over the `IntMap`. Here's what it will look like:

```Haskell
update g k = runPar $ do
    m <- Map.traverseWithKey (\i jmap -> spawn (return (shortmap i jmap))) g
    traverse get m
```

The call to `Map.traverseWithKey` gets us an `IntMap (IVar (IntMap Weight))`.
To get the new `Graph`, we call `get` on each of the `IVar`s and produce
a new `Graph` with all the elements using `traverse`.

### Pipeline Parallelism

**Pipeline Parallelism**:
the use parallelism between stages of a pipeline.

When a pipeline stage maintains some state,
we can't exploit parallelism as we do when we parallelize lazy streams.
Instead, we want each pipeline stage to run on a separate core,
with data streaming between them using the `Par` monad.

Instead of representing the stream as a lazy list,
we will use an explicit representation of a stream:

```Haskell
data IList a
    = Nil
    | Const a (IVar (IList a))

type Stream a = IVar (IList a)
```

An `IList` is a list with an `IVar` as a tail, allowing the producer
to generate the list incrementally, while a consumer runs in parallel,
grabbing elements as they are produced.

A `Stream` is an `IVar` containing an `IList`.

We need a few functions to work with `Streams`.
The first will be a generic producer that turns a lazy list into a `Stream`:

```Haskell
streamFromList :: NFData a => [a] -> Par (Stream a)
streamFromList xs = do
    var <- new -- creates the IVar that will be the Stream itself
    fork $ loop xs var -- forks the loop that will create the Stream contents
    return var -- returns the Stream to the caller. the Stream is now being
               -- created in parallel.
  where
    -- this loop traverses the input list, producing the IList as it goes
    -- the first argument is the list, and
    -- the second is the IVar into which to store the IList
    loop [] var = put var Nil -- empty list -> store an empty IList into the IVar
    loop (x:xs) var = do -- in the case of a non-empty list:
        tail <- new -- we create a new IVar for the tail
        put var (Cons x tail) -- and store a Cons cell representing the current elem
                              -- into the current IVar
        loop xs tail -- recurse to create the rest of the stream
```

Next, we'll write a consumer of `Streams`, `streamFold`:

```Haskell
-- this is pretty straightforward
streamFold :: (a -> b -> a) -> a -> Stream b -> Par a
streamFold fn !acc instrm = do
    ilst <- get instrm
    case ilst of
        Nil -> return acc
        Cons h t -> streamFold fn (fn acc h) t
```

The final is a map over `Streams`. This both a producer and a consumer:

```Haskell
-- also pretty straightforward
-- combo of patterns from streamFromList and streamFold
streamMap :: NFData b => (a -> b) -> Stream a -> Par (Stream b)
streamMap fn instrm = do
    outstrm <- new
    fork $ loop instrm outstrm
    return outstrm
  where
    loop instrm outstrm = do
        ilst <- get instrm
        case ilst of
            Nil -> put outstrm Nil
            Cons h t -> do
                newtl <- new
                put outstrm (Cons (fn h) newtl)
                loop t newtl
```

To demonstrate how this works, we'll create a pipeline with the previous RSA example.
The pipeline will be composing encrypt and decrypt (not realistic but example).

However, now they will work over `Stream ByteString` in the `Par` monad:

```Haskell
encrypt :: Integer -> Integer -> Stream ByteString -> Par (Stream ByteString)
encrypt n e s = streamMap (B.pack . show . power e n . code) s

decrypt :: Integer -> Integer -> Stream ByteString -> Par (Stream ByteString)
decrypt n d s = streamMap (B.pack . decode . power d n . integer) s
```

The following function composes them together (and does some other work):

```Haskell
pipeline :: Integer -> Integer -> Integer -> ByteString -> ByteString
pipeline n e d b = runPar $ do
    s0 <- streamFromList (chunk (size n) b)
    s1 <- encrypt n e s0
    s2 <- decript n d s1
    xs <- streamFold (\x y -> (y : x)) [] s2
    return (B.unlines (reverse xs))
```

#### Rate-Limiting the Producer

If the producer is faster than the consumer,
the producer would build up a lost `IList` chain in memory.
This large heap data structure would incur tons of overhead due to GC.
So we want to rate-limit the producer to avoid it getting too far ahead.

Here's a trick that adds some automatic rate-limiting to the stream API by
adding another constructor to the `IList` type. The creator will produce a fixed
amount of the list, then inserts a `Fork` constructor containing another `Par`
computation to produce more of the list:

```Haskell
data IList a
    = Nil
    | Cons a (IVar (IList a))
    | Fork (Par ()) (IList a)
```

#### Limitations of Pipeline Parallelism

Data parallelism (parallelized streaming from lazy list) exposes more parallelism
than pipeline parallelism because you can only create as much parallelism
as there are pipelines.
Still, pipeline parallelism is necessary when pipeline stages are stateful.

### Example: A Conference Timetable

This example is a program that finda a valid timetable for a conference.

* The conference runs *T* parallel tracks, and each track has the same number
  of talk slots, *S*; hence there are *T* * *S* talk slots in total.
* There are at most *T* * *S* talks to assign to tracks and slots (if there are
  fewer talks than slots, we can make up the difference with dummy talks)
* There are a number of attendees who have each expressed a preference for some
  talks that they would like to see
* The goal is to assign talks to slots and tracks so that the attendees can attend
  all the talks they want to see; that is, we never schedule two talks that an
  attendee wanted to see on two different tracks in the same slot

This is an example of a *constraint satisfaction* problem,
which means it requires an exhaustive search,
but we can be more clever than just generating all the possible assignments.
We can fill the timetable incrementally: assign a talk to the first slot of
the first track, then find a talk for hte first slot of the second track
that doesn't introduce a conflict, and so on.
Then we go to the second slot and so on until we've filled everything.
This avoids searching for solutions when the grid already contains a conflict.

If we can't fill a slot without conflicts, we must backtrack and choose a
different talk in the previous slot. If we can't in the previous slot, we must
backtrack even further. So, the search pattern is a tree.

Algorithms that have this tree-shape are called *divide and conquer* algorithms,
where the problem is recursively split into smaller subproblems solves separately,
then combined to form the whole solution.
Divide and conquer algorithms parallelize well because the branches are independent
of one another.

Let's code up a solution. Starting with a simple way to represent the types:

```Haskell
-- | Talk - simply numbered
newtype Talk = Talk Int
    deriving (Eq, Ord)

instance NFData Talk

instance Show Talk where
    show (Talk t) = show t

-- | Attendee - name and talks wanting to attend
data Person = Person
    { name :: String
    , talks :: [Talk]
    } deriving (Show)

-- | Complete Timetable -- Each list represents a single slot
type timeTable = [[Talk]]
```

The top-level function will have this signature:

```Haskell
timetable :: [Person] -> [Talk] -> Int -> Int -> [TimeTable]
timetable people allTalks maxTrack maxSlot =
```

First, we will cache some information about which talks clash.
We want to know which other talks cannot be scheduled in the same slot,
because one or more attendees want to see both of them.

```Haskell
-- note: selects takes a list and returns a list of pairs
-- 
-- ghci> selects [1..3]
-- [(1,[2,3]),(2,[1,3]),(3,[2,1])]

clashes :: Map Talk [Talk]
clashes = Map.fromListWith union
    [ (t, ts)
    | s <- people
    , (t, ts) <- selects talk s ]
```

Now we will write the algorithm itself. At each stage, we start with a partially
filled in timetable, and want to determine the possible ways of filling in the
next slot and generate all solutions from those.

```Haskell
generate :: Int -- current slot number
         -> Int -- current track number
         -> [[Talk]] -- slots allocated so far
         -> [Talk] -- talks in this slot
         -> [Talk] -- talks that can be allocated in this slot
         -> [Talk] -- all talks remaining to be allocated
         -> [TimeTable] -- all possible solutions
generate slotNo trakcNo slots slot slotTalks talks
    | slotNo == maxSlot = [slots]
    | trackNo == maxTrack =
        generate (slotNo+1) 0 (slot:slots) [] talks talks
    | otherwise = concat
        [ generate slotNo (trackNo+1) slots (t:slot) slotTalks' talks'
        | (t, ts) <- selects slotTalks
        , let clashesWithT = Map.findWithDefault [] t clashes
        , let slotTalks' = filter (`notElem` clashesWithT) ts
        , let talks' = filter (/= t) talks ]
```

Finally, we generate from the start:

```Haskell
generate 0 0 [] [] allTalks allTalks
```

#### Adding Parallelism

We'd prefer to separate the parallelism as far as possible from the algorithm code,
to avoid overcomplicating already quite involved code.

A lazy data structure doesn't work very well for this problem,
so Strategies would not be a good solution.

Instead, we'll build a *parallel skeleton*:
a higher-order function that abstracts a pattern of computation.

We will create a `search` skeleton:

```Haskell
search :: ( partial -> Maybe solution ) -- finished?
       -> ( partial -> [ partial ] ) -- refine a solution
       -> partial -- initial, empty partial
       -> [solution]
search finished refine emptysoln = generate emptysoln
  where
    generate partial
        | Just soln <- finished partial = [soln]
        | otherwise = concat (map generate (refine partial))
```

Now to refactor timetable to use `search`:

```Haskell
type Partial = (Int, Int, [[Talk]], [Talk], [Talk], [Talk])

timetable :: [Person] -> [Talk] -> Int -> Int -> [TimeTable]
timetable people allTalks maxTrack maxSlot =
    search finished refine emptysoln

  where
    emptysoln = (0, 0, [], [], allTalks, allTalks)

    finished (slotNo, trackNo, slots, slot, slotTalks, talks)
        | slotNo == maxSlot = Just slots
        | otherwise = Nothing

    clashes :: Map Talk [Talk]
    clashes = Map.fromListWith union
        [ (t, ts)
        | s <- people
        , (t, ts) <- selects (talks s)]

    refine (slotNo, trackNo, slots, slot, slotTalks, talks)
        | trackNo == maxTrack = [(slotNo+1, 0, slot:slots, [], talks, talks)]
        | otherwise =
            [ (slotNo, trackNo+1, slots, t:slot, slotTalks', talks')
            | (t, ts) <- selects slotTalks
            , let clashesWithT = Map.findWithDefault [] t clashes
            , let slotTalks' = filter (`notElem` clashesWithT) ts
            , let talks' = filter (/= t) talks ]
```

This works exactly the same way, we just now need to parallelize the search skelly.
At each stage, we'll spawn off the recursive calls in parallel and
then collect the results.

```Haskell
parsearch :: NFData solution
          => (partial -> Maybe solution)
          -> (partial -> [partial])
          -> partial
          -> [solution]
parsearch finished refine emptysoln
    = runPar $ generate emptysoln
  where
    generate partial
        | Just soln <- finished partial = return [soln]
        | otherwise = do
            solnss <- parMapM generate (refine partial)
            return (concat solnss)
```

`parMapM` calls generate in parallel on the list of partial solutions.
The granularity gest too fine as we get near the leaves of the search tree,
so it adds too much parallelization overhead in relation to the computational
power required for those leaves.

We can't chunk a tree, so we will add a *depth threshold*.
Spawn recursive calls in parallel down to a certain depth,
then just use the sequential algorithm.

Let's add a depth threshold to parsearch:

```Haskell
parsearch :: NFData solution
        => Int -- depth threshold
        -> (partial -> Maybe solution)
        -> (partial -> [partial])
        -> partial
        -> [solution]
parsearch maxdepth finished refine emptysoln
    = runPar $ generate 0 emptysoln
  where
    generate d partial | d >= maxdepth
        = return (search finished refine partial)
    generate d partial
        | Just soln <- finished partial = return [soln]
        | otherwise = do
            solnss <- parMapM (generate (d+1)) (refine partial)
            return (concat solnss)
```

This gives us a better speedup.
Also, shows the benefit of the skeleton:
the modularity allowed us to easily add the depth threshold.

#### Example Takeaways

* Tree-shaped (*divide and conquer*) computations parallelize well
* You can abstract the parallel pattern as a *skeleton* using higher-order functions
* To control the granularity in a tree-shaped computation,
  add a *depth threshold* and use the sequential version below a certain depth

### Example: A Parallel Type Inferencer

Given a list of bindings of the form `x = e` for a variable `x` and expression `e`,
infer the types for each of the variables.
Expressions consist of integers, variables, application, lambda expressions,
`let` expressions, and arithmetic operators.

We will be adding parallelization to an old inference engine.
The types from it are as follows:

```Haskell
type VarId = String -- Variables
data Term -- Terms in the input program
data Env -- Environment, mapping VarId to PolyType
data PolyType -- Polymorphic types
```

Then we have a few function signatures:

```Haskell
-- create an env
makeEnv :: [(VarId,PolyType)] -> Env

-- extract unbound variables of an expression
freeVars :: Term -> [VarId]

-- deliver a PolyType based on the terms and environment
inferTopRhs :: Env -> Term -> PolyType
```

The sequential part of the engine uses an `Env` that maps `VarIds` to `PolyTypes`.
the parallel part of the engine will use an environment that maps `VarIds` to
`IVar PolyType`, so that we can fork the inference engine for a given binding,
then wait for its result later. The environment for the parallel type inferencer
is called `TopEnv`:

```Haskell
type TopEnv = Map VarId (IVar PolyType)
```

All that remains is to write the top-level loop. We'll do this in two stages.
First, a function to infer the type of a single binding:

```Hskell
inferBind :: TopEnv -> (VarId,Term) -> Par TopEnv
inferBind topenv (x,u) = do
    vu <- new
    fork $ do
        let fu = Set.toList (freeVars u)
        tfu <- mapM (get . fromJust . flip Map.lookup topenv) fu
        let aa = makeEnv (zip fu tfu)
        put vu (inferTopRhs aa u)
    return (Map.insert x vu topenv)
```

This creates an `IVar`, `vu`, to hold the type of this binding.

Then forks the computation that does the type inference.

The inputs to this type inference are the types of the variables mentioned
in the expression `u`. Hence we call `freeVars` to get those variables.

For each of the free variables, we look up its `IVar` in the `topenv`,
and then call `get` on it. Hence this step will wait until the types of all the
free variables are available before proceeding.

Then we build an `Env` from the free variables and their types.

Then we infer the type of the expression `u`, and
put the result in the `IVar` we created at the beginning.

Back in the parent, we return `topenv` extended with `x` mapped to
the new `IVar` `vu`.

Next, we use `inferBind` to define `inferTop`,
which infers types for a list of bindings:

```Haskell
inferTop :: TopEnv -> [(VarId, Term)] -> Par [(VarId,Polytype)]
inferTop topenv0 binds = do
    topenv1 <- foldM inferBind topenv0 binds
    mapM (\(v, i) -> do t <- get i; return (v, t)) (Map.toList topenv1)
```

Here, we use `foldM` to perform `inferBind` over each binding,
accumulating a `TopEnv` that will contain a mapping for each of the variables.
Then, we wait for all the type inference to happen, and collect the results.
Hence, we turn the `TopEnv` back into a list and call `get` on all the `IVars`.

### Using Different Schedulers

We can change the scheduling strategy used in the `Par` monad
without changing GHC or its runtime system since its implemented as a library.

There are two schedulers in the `monad-par` library: Trace and Direct.
Trace is usually slightly worse, but not always, so its worth trying it out.

To use the Trace scheduler instead of the Direct scheduler:

```Haskell
import Control.Monad.Par.Scheds.Trace
-- instead of Control.Monad.Par
```

This change will need to be made in all the modules that import `Control.Monad.Par`

### The Par Monad Compared to Strategies

Both are generally effective, so sometimes just a matter of preference.
However, bear in mind these tradeoffs:

* If the algorithm naturally produces a lazy data structure, Strategies will work well.
  If not, then the Par monad might be more straightforward to express parallelism.
* `runPar` is relatively expensive, wherease `runEval` is free.
  So try to thread the `Par` monad around to all the places that need parallelism
  to avoid needing multiple `runPar` calls.
  If this is inconvenient, `Eval` or Strategies might be a better choice.
* Strategies allow a separation between algorithm and parallelism.
  However, a parallel skeleton works with both approaches.
* `Par` has more overhead than `Eval`,
  so `Eval` performs better at finer granularities
  (at larger granularities, performance is about the same).
* `Par` is implemented entirely as a library, so it's easily modified.
* `Eval` has more diagnostics in ThreadScope.
* `Par` does not support speculative parallelism:
  parallelism in the `Par` monad is always executed.