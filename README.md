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

```shell
ghci> let x = 1 + 2 :: Int
ghci> :sprint x
x = _
```

An object in memory representing an unevaluated computation is called a *thunk*.
Here, x is a pointer to an object in memory representing the function `+`
applied to the integers `1` and `2`.

Let's look at another example:

```shell
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

## Concurrent Haskell

GHC provides lightweight threads.
The built-in functionality is sparse, but general so we can implement our own abstractions

Therefore, we will learn the low-level features and build up higher-level abstractions
to learn Concurrent Haskell.

## Basic Concurrency: Threads and MVars

The fundamental action in concurrency is forking a new thread of control:

```Haskell
forkIO :: IO () -> IO ThreadId
```

This operation takes in the type `IO ()`:
a computation in the `IO` monad that eventually delivers a value of type `()`.
This computation is executed in a new concurrent thread.
If the thread has effects, they will be interleaved indeterminately with
the effects of other threads.

In Haskell, the program terminates when `main` returns,
even if other threads are still running.

### Communication: MVars

The API for `MVar` is as follows:

```Haskell
data MVar a -- abstract

newEmptyMVar :: IO (MVar a)
newMVar :: a -> IO (MVar a)
takeMVar :: MVar a -> IO a
putMVar :: MVar a -> a -> IO ()
```

An `MVar` can be thought of like a box.
We have a function that creates an empty box,
and a general one that creates a full box containing the value passed as argument.
`takeMVar` removes the value from the box and returns it,
but waits if the box is currently empty.
Symmetrically, `putMVar` puts a value into the `MVar`,
but waits if the box is currently full.

This is the fundamental building block that generalizes many different
communication and synchronization patterns. Here are the main ways it can be used:

* passing messages between threads, holding one message at a time, since its a
  *one-place channel*
* as a container for shared mutable state, taking the current value,
  then placing a new value back in it
* as a building block for larger concurrent data structures

### MVar as a Simple Channel: A Logging Service

Our logging service will have the following API:

```Haskell
data Logger

initLogger :: IO Logger
logMessage :: Logger -> String -> IO ()
logStop :: Logger -> IO ()
```

There's an abstract data type that represents a handle to the logging service,
and a new logging service is created with `initLogger`.
Having `Logger` be a value that we pass around (rather than a top-level global)
is good practice, it means we could have multiple loggers for example.

Then we have the two operations. The stop operation is important because we want
to make sure all messages have been logged before we terminate the program.

Let's implement incrementally:

```Haskell
data Logger = Logger (MVar LogCommand)

data LogCommand = Message String
                | Stop (MVar ())
```

The `Logger` is just an `MVar` that we use as a channel for communication with
the logging thread.

There are two kinds of requests we can make, so `LogCommand` takes either the
straightforward `Message` or a `Stop` that contains `MVar ()` to enable the sender
of the stop request to wait for a reply from the logging thread that indicates
its finished.

Now let's implement initialization, where we create an empty `MVar` for
the channel and fork a thread to perform the service (the `logger` function):

```Haskell
initLogger :: IO Logger
initLogger = do
    m <- newEmptyMVar
    let l = Logger m
    forkIO (logger l)
    return l

logger :: Logger -> IO ()
logger (Logger m) = loop
  where
    loop = do
        cmd <- takeMVar m
        case cmd of
            Message msg -> do
                putStrLn msg
                loop
            Stop s -> do
                putStrLn "logger: stop"
                putMVar s ()
```

The `logger` function recursively retrieves the next `LogCommand` from the `MVar`.
If it is a message, it just prints it and recurses.
If it is a stop command, it puts the unit value into the `MVar` from the command,
then returns, causing the logger thread to exit.

The implementation of the rest of the functions are trivial:

```Haskell
logMessage :: Logger -> String -> IO ()
logMessage (Logger m) s = putMVar m (Message s)

logStop :: Logger -> IO ()
logStop (Logger m) = do
    s <- newEmptyMVar
    putMVar m (Stop s)
    takeMVar s
```

`logStop` creates an `MVar` to hold the response, then sends a stop command to the
logger containing the new empty `MVar`. After sending the command, we call `takeMVar`
on the new `MVar` to wait for the response.

Here's an trivial example:

```Haskell
main :: IO ()
main = do
    l <- initLogger
    logMessage l "hello"
    logMessage l "bye"
    logStop l
```

### MVar as a Container for Shared State

Concurrent programs often share mutable state that we usually need to perform
operations on. We need to be able to perform these operations in a way
that appears atomic from the POV of other threads (other threads should not
be able to view intermediate states, nor be able to initiate their own
operations while another operation is in progress).

`MVar` can act as a sort of lock, where `takeMVar` acquires the lock,
and `putMVar` releases it.

Let's go through an example of a phonebook as a piece of mutable state:

```Haskell
type Name = String
type PhoneNumber = String
type PhoneBook = Map Name PhoneNumber

newtype PhoneBookState = PhoneBookState (MVar PhoneBook)

new :: IO PhoneBookState
new = do
    m <- newMVar Map.empty
    return (PhoneBookState m)

insert :: PhoneBookState -> Name -> PhoneNumber -> IO ()
insert (PhoneBookState m) name number = do
    book <- takeMVar m
    putMVar m (Map.insert name number book)
```

This shows an important principle:
we can take *any* pure immutable data structure such as `Map` and
turn it into mutable shared state simply by wrapping in an `MVar`

It's important to understand the effect of laziness in this line:

```Haskell
putMVar m (Map.insert name number book)
```

This places in the `MVar` the *unevaluated* expression.
This can be good, because then we only hold the lock very briefly.
However, if we were to do many `insert` operations consecutively,
the `MVar` would build up a large chain of unevaluated expressions,
creating a space leak. As an alternative, we could force evaluation:

```Haskell
putMVar m $! Map.insert name number book
```

To get brief locking *and* no space leaks, we can do this:

```Haskell
let book' = Map.insert name number book
putMVar m book'
seq book' (return ())
```

This stores the unevaluated expression in the `MVar`,
then immediately evaluates it. So, the lock is held only briefly, and
the thunk is evaluated.

### MVar as a Building Block: Unbounded Channels

Let's build a larger abstraction with `MVar`:
an unbounded buffer channel. Here's the interface:

```Haskell
type Stream a = MVar (Item a)
data Item a   = Item a (Stream a)

data Chan a
  = Chan (MVar (Stream a))
         (MVar (Stream a))

newChan :: IO (Chan a)
newChan = do
  hole  <- newEmptyMVar
  readVar  <- newMVar hole
  writeVar <- newMVar hole
  return (Chan readVar writeVar)

writeChan :: Chan a -> a -> IO ()
writeChan (Chan _ writeVar) val = do
  newHole <- newEmptyMVar
  oldHole <- takeMVar writeVar
  putMVar oldHole (Item val newHole)
  putMVar writeVar newHole

readChan :: Chan a -> IO a
readChan (Chan readVar _) = do
  stream <- takeMVar readVar
  Item val tail <- readMVar stream
  putMVar readVar tail
  return val
```

Programming larger structures with `MVar` can be pretty tricky to reason about
and avoid deadlocks/race conditions (this is where STM comes in).

## Overlapping Input/Output

Let's look at an exapmle that asynchronously downloads files from a URL:

```Haskell
sites :: [String]

data Async a = Async (MVar a)

async :: IO a -> IO (Async a)
async action = do
    var <- newEmptyMVar
    forkIO (do r <- action; putMVar var r)
    return (Async var)

wait :: Async a -> IO a
wait (Async var) = readMVar var

main = do
    as <- mapM (async . getURL) sites
    rs <- mapM wait as
    print rs
```

This example ignores exceptions and error handling. Let's investigate how that works.

### Exceptions in Haskell

Haskell has no built-in semantics for exception handling;
everything is done with library functions.

Exceptions are thrown with this function:

```Haskell
throw :: Exception e => e -> a
```

Note that it takes any type that is an instance of the `Exception` type class and
returns the unrestricted type variable `a`, so it can be called from anywhere.

The `Exception` type class is defined as follows:

```Haskell
class (Typeable e, Show e) => Exception e where
    -- ...
```

So, any type that is an instance of both `Typeable` and `Show` can be an `Exception`.

One common type used as an exception is `ErrorCall`:

```Haskell
newtype ErrorCall = ErrorCall String
    deriving (Typeable)

instance Show ErrorCall where {...}

instance Exception ErrorCall

-- now we can throw an ErrorCall like so
throw (ErrorCall "oops!")
```

The function `error` from the prelude does exactly this:

```Haskell
error :: String -> a
error s = throw (ErrorCall s)
```

I/O operations throw exceptions to indicate errors, usually `IOException` type.
Operations for `IOException` are in `System.IO.Error` library.

Exceptions can be caught, but only in the IO monad:

```Haskell
catch :: Exception e => IO a -> (e -> IO a) -> IO a
```

The two arguments are: the IO operation to perform and
an exception handler of type `e -> IO a` where `e` must be an instance of
the `Exception` class.

The IO operation in the first argument is performed, and if it throws an exception
of the type expected by the handler, `catch` executes the handler,
passing in the exception value that was thrown.

Sometimes it is more convenient to use the `try` variant:

```Haskell
try :: Exception e => IO a -> IO (Either e a)
```

For example,

```shell
> try (readFile "nonexistent") :: IO (Either IOException String)
Left nonexistent: openFile: does not exist (No such file or directory)
```

Another variant of `catch` is `handle`, which is just `catch` with args reversed:

```Haskell
handle :: Exception e => (e -> IO a) -> IO a -> IO a
```

This is useful when the exception handler is short, but the action is long:

```Haskell
handle (\e -> ...) $ do
    ...
```

We can also perform some operation if an exception is raised and
then re-throw the exception with `onException`:

```Haskell
onException :: IO a -> IO b -> IO a
onException io what
    = io `catch` \e -> do _ <- what
                          throwIO (e :: SomeException)
```

Last is two useful higher-level abstractions, `bracket` and `finally`:

```Haskell
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c

finally :: IO a -> IO b -> IO a
```

`bracket` allows us to set up an exception handler to deallocate a resource or
perform some sort of cleanup operation. For example, suppose we want to create a
temporary file, perfom some operation, and have the temporary file reliable
removed afterward - even if an exception occured during the operation.
We would use `bracket`:

```Haskell
bracket (newTempFile "temp") -- allocates resource, result used as arg for next two
        (\file -> removeFile file) -- deallocates resource
        (\file -> ...) -- the operation to perform
```

`bracket` can be defined like so:

```Haskell
bracket IO a -> (a -> IO b) -> (a -> IO c) -> IO c
bracket before after during = do
    a <- before
    c <- during a `onException` after a
    after a
    return c
```

`finally` is a special case of `bracket` that can be defined like so:

```Haskell
finally :: IO a -> IO b -> IO a
finally io after = do
    io `onException` after
    after
```

### Error Handling with Async

To handle errors in our last example, we will need to handle when the `Async`
operation throws an exception by wrapping the action in `try`:

```Haskell
data Async a = Async (Mvar (Either SomeException a))

async :: IO a -> IO (Async a)
async action = do
    var <- newEmptyMVar
    forkIO (do r <- try action; putMVar var r)
    return (Async var)

waitCatch :: Async a -> IO (Either SomeException a)
waitCatch (Async var) = readMVar var

wait :: Async a -> IO a
wait a = do
    r <- waitCatch a
    case r of
        Left e -> throwIO e
        Right a -> return a
```

`waitCatch` allows the caller to handle the error immediately,
while `wait` rethrows the error to propogate it.

### Merging

Suppose we want to wait for one of several events to occur. For example,
say we just want to wait for one of sites to complete downloading.

We could modify the `getURL` example to feed all the results into the same
`MVar` like this:

```Haskell
sites :: [String]

main :: IO ()
main = do
    m <- newEmptyMVar
    let
        download url = do
            r <- getURL url
            purMVar m (url, r)

    mapM_ (forkIO . download) sites

    (url, r) <- takeMVar m
    printf "%s was first (%d bytes)" url (B.length r)
    replicateM_ (length sites - 1) (takeMVar m)
```

This puts all the results in the same `MVar`, but only prints the first one,
then waits for the rest to complete before the program is done executing,
telling us which was the quickest to download.

This can be a bit inconvenient to arrange so all events feed to the same `MVar`,
lets extend our Async API to allow waiting for two Asyncs simultaneously:

```Haskell
waitEither :: Async a -> Async b -> IO (Either a b)
wait Either a b = do
    m <- newEmptyMVar
    forkIO $ do r <- try (fmap Left (wait a)); putMVar m r
    forkIO $ do r <- try (fmap Right (wait b)); putMVar m r
    wait (Async m)
```

We can generalize this to wait for a list of Asyncs:

```Haskell
waitAny :: [Async a] -> IO a
waitAny as = do
    m <- newEmptyMVar
    let forkwait a = forkIO $ do r <- try (wait a); putMVar m r
    mapM_ forkwait as
    wait (Async m)
```

Now, we can simplify the changes to the `main` function:

```Haskell
main :: IO
main = do
    let
        download url = do
            r <- getURL url
            return (url, r)

    as <- mapM (async . download) sites

    (url, r) <- waitAny as
    printf "%s was first (%d bytes)\n" url (B.length r)
    mapM_ wait as
```

(STM will later solve the issue of needing to create a new thread
for every operation in this example)

## Cancellation and Timeouts

It's often important for one thread to be able to interrupt the execution of
another thread. Such as a user closing the application, a timeout, or
changing the parameters of a long compute-intensive task as its executing.

The design decision is whether or not the intended victim thread should
have to poll for the cancellation condition or whether the thread is immediately
cancelled in some way. This is a tradeoff:

1. Possible that polling is not regular, and the thread will become unresponsive,
   leading to deadlocks and hangs.
2. Asynchronous cancellation calls for critical sections that modify state
   needing to be protected from cancellation. Otherwise, cancellation may occur
   mid-update, leaving some data in an inconsistent state.

Fully asynchronous cancellation is the default in Haskell since most code
is purely functional able to be safely suspended and unable to poll.
Therefore, the design decision reduces to deciding how cancellation is handled
by code in the `IO` monad.

### Asynchronous Exceptions

An **asynchronous exception** is asynchronous from the point of view of the victim;
they didn't ask for it and it can arrive at any time.

Conversely, a **synchronous exception** are thrown using the normal
`throw` and `throwIO`.

To initiate an asynchronous exception, we use `throwTo`,
which throws an exception from one thread to another:

```Haskell
throwTo :: Exception e => ThreadId -> e -> IO ()
```

The exception must be an instance of the `Exception` class. The `ThreadId` is
returned by a previous call to `forkIO` and may refer to a thread in any state.

Let's look at an example of asynchronous exceptions in our Async API:

```Haskell
-- this will cancel an existing Async
cancel :: Async a -> IO

-- we will need to store the ThreadId of the thread running Async for this to work
data Async a = Async ThreadId (MVar (Either SomeException a))

-- here's how we will implement cancel with these changes
cancel (Async t var) = throwTo t ThreadKilled

-- the async operator will need to store the ThreadId, too
async :: IO a -> IO (Async a)
async action = do
    m <- newEmptyVar
    t <- forkIO (do r <- try actoin; putMVar m r)
    return (Async t m)

-- now we're ready to update main to support quitting
main = do
    as <- mapM (async . timeDownload) sites

    forkIO $ do -- new thread to poll for quit
        hSetBuffering stdin NoBuffering
        forever $ do
            c <- getChar
            when (c == 'q') $ mapM_ cancel as

    rs <- mapM waitCatch as
    printf "%d/$d succeeded\n" (length (right rs)) (length rs)
```

### Masking Asynchronous Exceptions

We need a way to control the delivery of asynchronous exceptions in case
the exception is made while the victim thread is updating some shared state.

If a thread wishes to take an `MVar`, perform an operation depending on
the value in it, then put the result back in it, the code must be responsive
to asynchronous exceptions, but it should be safe.
If an exception arrives after the `takeMVar` but before the final `putMVar`,
the `MVar` should not be left empty, the original value should be restored.

Here's the problem:

```Haskell
problem :: MVar a -> (a -> IO a) -> IO ()
problem m f = do
    a <- takeMVar m
    r <- f a `catch` \e -> do putMVar m a; throw e
    putMVar m r
```

If the exception hits between any of the lines of the function,
the invariant will be violated. To fix this, we must use the `mask` combinator:

```Haskell
mask :: ((IO a -> IO a) -> IO b) -> IO b
```

`mask` defers the delivery of asynchronous exceptions for the duration of its argument.
Let's look at an example:

```Haskell
problem :: MVar a -> (a -> IO a) -> IO ()
problem m f = mask $ \restore -> do
    a <- takeMVar m
    r <- restore (f a) `catch` \e -> do putMVar m a; throw e
    putMVar m r
```

`mask` is applied to a function, which takes as its argument a `restore` function.
Now, exceptions can only be raised while `(f a)` is working, and we have an
exception handler to catch exceptions in that case.

We can provie higher-level combinators to insulate programmers from
the need to use mask directly. For example, `problem` is given as
`modifyMVar_` in the `Control.Concurrent.MVar` library:

```Haskell
modifyMVar_ :: MVar a -> (a -> IO a) -> IO ()

-- also a variant that allows the operation to return a separate result in addition
-- to the new contents of the MVar
modifyMVar :: MVar a -> (a -> IO (a, b)) -> IO b
```

Here's a simple example of `modifyMVar`, used to implement compare and swap:

```Haskell
casMVar :: Eq a => MVar a -> a -> a -> IO Bool
casMVar m old new =
    modifyMVar m $ \cur ->
        if cur == old
           then return (new,True)
           else return (cur,False)
```

This function returns the the new value and `True` if the `MVar` contents are old,
otherwise it returns the current contents and `False`.

Here's an example working on multiple `MVar`s by nesting calls to `modifyMVar`:

```Haskell
modifyTwo :: MVar a -> MVar b -> (a -> b -> IO (a, b)) -> IO ()
modifyTwo ma mb f =
    modifyMVar_ mb $ \b ->
        modifyMVar ma $ \a -> f a b
```

### The `bracket` Operation

`bracket` is actually defined with `mask` to make it safe in the presence
of asynchronous exceptions:

```Haskell
bracket :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c
bracket before after thing =
    mask $ \restore -> do
        a <- before
        r <- restore (thing a) `onException` after a
        _ <- after a
        return r
```

`before` and `after` are performed inside a `mask`.

### Asynchronous Exception Safety for Channels

In most `MVar` code, we can use operations like `modifyMVar` insteadof `takeMVar`
and `putMVar` to make our code safe in the presence of asynchronous exceptions.

For example, here's how we can prevent deadlocks from asynchronous exceptions
in `readChan`:

```Haskell
-- old impl
readChan :: Chan a -> IO a
readChan (Chan readVar _) = do
    stream <- takeMVar readVar
    Item val new <- readMVar stream
    putMVar readVar new
    return val

-- new impl
readChan :: Chan a -> IO a
readChan (Chan readVar _) = do
    modifyMVar readVar $ \stream -> do
        Item val tail <- readMVar stream
        return (tail, val)
```

Now let's do the same thing for `writeChan`:

```Haskell
-- old impl
writeChan :: Chan a -> a -> IO ()
writeChan (Chan _ writeVar) val = do
    newHole <- newEmptyMVar
    oldHole <- takeMVar writeVar
    putMVar oldHole (Item val newHole)
    putMVar writeVar newHole

-- first thought
writeChan :: Chan a -> a -> IO ()
writeChan (Chan _ writeVar) val = do
    newHole <- newEmptyMVar
    modifyMVar_ writeVar $ \oldHole -> do
        putMVar oldHole (Item val newHole)
        return newHole
-- this wont work because an exception could strike between putMVar and return

-- heres the proper impl
writeChan :: Chan a -> a -> IO ()
writeChan (Chan _ writeVar) val = do
    newHole <- newEmptyMVar
    mask_ $ do
        oldHole <- takeMVar writeVar
        putMVar oldHole (Item val newHole)
        putMVar writeVar oldHole
```

Note that the two `putMVar`s are guaranteed not to block,
so they are not interruptible.

### Timeouts

To write a timeout function, we fork a new thread that will wait for
`t` microseconds and then call `throwTo` to throw the `Timeout` exception
back to the original thread. If the operatoin completes within the time limit,
then we must ensure that this thread never throws its `Timeout` exception,
so `timeout` must kill the thread before returning.

```Haskell
timeout :: Int -> IO a -> IO (Maybe a)
timeout t m
    | t < 0 = fmap Just m
    | t == 0 = return Nothing
    | otherwise = do
        pid <- myThreadId
        u <- newUnique
        let ex = Timeout u
        handleJust
            (\e -> if e == ex then Just () else Nothing)
            (\_ -> return Nothing)
            (bracket (forkIO $ do threadDelay t
                                  throwTo pid ex)
                     (\tid -> throwTo tid ThreadKilled)
                     (\_ -> fmap Just m)
```

### Catching Asynchronous Exceptions

Async exceptions propogate like normal exceptions but need special handling.

Haskell automatically masks async exceptions inside exception handlers.

Main pitfall: tail-calling from handlers keeps programs masked accidentally.

Use `try` instead of `catch`/`handle` when possible to avoid masking issues.

Keep exception handling scope narrow and avoid nesting handlers.

For example:

```Haskell
-- BAD (stays masked)
main = do
    fs <- getArgs
    let
        loop n! [] = return n
        loop n! (f:fs)
            = handle (\e -> if isDesNotExistError e
                               then loop n fs -- tail call keeps mask
                               else throwIO e) $
                do
                    h <- openFile f ReadMode
                    s <- hGetContents h
                    loop (n + length (lines s)) fs
    n <- loop 0 fs
    print n

-- GOOD (proper masking)
main = do
    fs <- getArgs
    let
        loop !n [] = return n
        loop !n (f:fs) = do
            getMaskingState >>= print
            r <- Control.Exception.try (openFile f ReadMode)
            case r of
                Left e | isDoesNotExistError e -> loop n fs
                       | otherwise -> throwIO e
                Right h -> do
                    s <- hGetContents h
                    loop (n + length (lines s)) fs
    n <- loop 0 fs
    print n
```

The good version avoids accidental masking and keep exception handling local.

### mask and forkIO

`forkIO` creates threads that inherent parent's masking state.

Need to handle exceptions in window between thread creation and exception handling.

`forkFinally` provides safer pattern for common "do after thread completion" cases.

Rule of thumb: if first hing in `forkIO` is exception handling, use `forkFinally`.

Example:

```Haskell
-- BAD (has race condition)
async :: IO a -> IO (Async a)
async action = do
    m <- newEmptyMVar
    t <- forkIO (do r <- try action; putMVar m r) -- vulnerable window
    return (Async t m)

-- GOOD (safe)
async action = do
    m <- newEmptyMVar
    t <- forkFinally action (putMVar m) -- handles completion safely
    return (Async t m)
```

The good version uses `forkFinally` to safely handle thread completion
and eliminate race conditions in exception handling.

### Asynchronous Exceptions: Discussion

Dealing with asynchronous exceptions at this level is something
Haskell programmers rarely have to do:

* All non-IO Haskell code is automatically safe by construction.
  This makes asynchronous exceptions feasible.
* We can use provided abstractions like `bracket` to acquire and release resources.
  These have safety built in.
  When working with `MVar`s, the `modifyMVar` family of ops also provides safety.

Making most IO monad code safe is straightforward, but for those
cases where things get complicated, a few techniques simplify things:

* Large chunks of heavily stateful code can be wrapped in a `mask`,
  which drops into polling mode for asynchronous exceptions.
* Using STM instead of `MVar`s or other state representations can sweep away all
  the complexity in one go.

## Software Transactional Memory

**Software Transactional Memory (STM)**:
a technique for simplifying concurrent programming by allowing
multiple state-changing operations to be groups together and performed as
a single atomic operation.

Here's the STM interface:

```Haskell
data STM a -- abstract
instace Monad STM -- among other things

atomically :: STM a -> IO a

data TVar a --abstract
newTVar :: a -> STM (TVar a)
readTVar :: TVar a -> STM a
wrteTVar :: TVar a -> a -> STM ()

retry :: STM a
orElse :: STM a -> STM a -> STM a

throwSTM :: Exception E => e -> STM a
catchSTM :: Exception E => STM a -> (e -> STM a) -> STM a
```

### Running Example: Managing Windows

Imagine a WM where the user can move windows from one desktop to another,
while at the same time, a program can request that its own window move from its
current desktop to another desktop. The WM uses multiple threads:
one to listen for input from the user, a set of threads to listen for requests
from programs running in each existing window, and one thread that renders the display.

Let's assume some abstract types:

```Haskell
data Desktop -- abstract
data Window -- abstract

type Display = Map Desktop (MVar (Set Window))
```

We are bound to run into Dining Philosopher's problem when moving windows if
there are converse `moveWindow` calls.

STM provides a way to avoid this deadlock problem without imposing any
requirements on the programmer. Lets replace `MVar` with `TVar`:

```Haskell
type Display = Map Desktop (TVar (Set Window))
```

A `TVar` is a transactional variable, a mutable variable that can be read or
written only within the `STM` monad using `readTVar` and `writeTVar`.

A computation in the `STM` monad can be performed in the `IO` monad,
using the `atomically` function.

When an `STM` computation is performed like this, its called a transaction
because the whole operation takes place atomically with respect to the rest
of the program - no other thread can observe an intermediate state.

Let's implement `moveWindow` using `STM`:

```Haskell
moveWindowSTM :: Display -> Window -> Desktop -> Desktop -> STM ()
moveWindowSTM disp win a b = do
    wa <- readTVar ma
    wb <- readTVar mb
    writeTVar ma (Set.delete win wa)
    writeTVar mb (Set.insert win wb)
  where
    ma = disp ! a
    mb = disp ! b
```

Then, we wrap this in `atomically` to make the `IO`-monad version `moveWindow`:

```Haskell
moveWindow :: Display -> Window -> Desktop -> Desktop -> IO ()
moveWindow disp win a b = atomically $ moveWindowSTM disp win a b
```

STM is far less error-prone here, and it scales to any number of `TVar`s.

Now suppose, we want to swap two windows,
moving window *W* from desktop *A* to *B*, and simulatenously *V* from *B* to *A*.
We can do this neatly with STM by simply making two calls to `moveWindowSTM`:

```Haskell
swapWindows :: Display
            -> Window -> Desktop
            -> Window -> Desktop
            -> IO ()
swapWindows disp w a v b = atomically $ do
    moveWindowsSTM disp w a b
    moveWindowsSTM disp v b a
```

This shows the composability of STM operations:
any operation of type `STM a` can be composed with others to form a
larger atomic transaction. For this reason, `STM` operations are usually provided
without the `atomically` wrapper so that we can compose them as necessary
before finally wrapping everything in `atomically`.

### Blocking

STM uses `retry` to deal with *blocking* (when we need to wait for
some condition to be true). `retry` tells STM computation to abandon the transaction
and try again.

Let's consider how to implement `MVar` using `STM` becuase `takeMVar` and
`putMVar` need to be able to block when the `MVar` is empty or full.

First the data type: an `MVar` is always in one of two states: full or empty:

```Haskell
newtype TMVar a = TMVar (Tvar (Maybe a))
```

To make an empty `TMVar`, we simply need a `TVar` containing `Nothing`:

```Haskell
newEmptyTMVar :: STM (TMVar a)
newEmptyTMVar = do
    t <- newTVar Nothing
    return (TMVar t)
```

Now, to take `TMVar`s, we need to block if the desired variable is empty and
return the content once the variable is set:

```Haskell
takeTMVar :: TMVar a -> STM a
takeTMVar (TMVar t) = do
    m <- readTVar t
    case m of
        Nothing -> retry
        Just a -> do
            writeTVar t Nothing
            return a
```

`putTMVar` is pretty straightforward:

```Haskell
putTMVar :: TMVar a -> a -> STM ()
putTMVar (TMVar t) a = do
    m <- readTVar t
    case m of
        Nothing -> do
            writeTVar t (Just a)
            return ()
        Just _ -> retry
```

Now, we can compose these `STM` operations:

```Haskell
atomically $ do
    a <- takeTMVar ta
    b <- takeTMVar tb
    return (a,b)
```

Operating on two `MVar`s is more contrived, since taking a single one is a side-effect
visible to the rest of the program, it can't be undone if the second `MVar` is empty
but with `STM` we can continue to `retry` the entire transaction.

### Blocking Until Something Changes

`retry` allows us to block on arbitrary conditions.

For example, say we want to render only the focused desktop
while windows may move around and appear or disappear on their own accord,
we would need to make sure the rendering thread updates accordingly.

Let's define `render` and `getWindows` as helpers, then we'll see how
`retry` blocks until something changes in the example:

```Haskell
-- handles rendering windows on display
-- should be called whenever the layout changes
render :: Set Window -> IO ()

-- returns the set of windows to render
-- given the Display and UserFocus
getWindows :: Display -> UserFocus STM (Set Window)
getWindows disp focus = do
    desktop <- readTVar focus
    readTVar (disp ! desktop)

-- use retry to avoid calling render when nothing has changed
renderThread :: Display -> UserFocus -> IO ()
renderThread disp focus = do
    wins <- atomically $ getWindows disp focus
    loop wins
  where
    loop wins = do
        render wins
        next <- atomically $ do
            wins' <- getWindows disp focus
            if (wins == wins')
                then retry
                else return wins'
        loop next
```

In `renderThread`, `retry` waits until the value read by `getWindows` could possibly
be different (another thread completing a transaction that writes to one of the
`TVar`s used by `getWindows`).

Thanks to STM, we don't have to implement this complex logic ourselves,
and can avoid many self-inflicted concurrency errors.

### Merging with STM

```Haskell
orElse :: STM a -> STM a -> STM a
```

The operation `orElse a b` has the following behavior:

* `a` is executed. If `a` has a result, the it's returned and `orElse` ends
* If `a` calls `retry` instead, `a` is discarded and `b` is executed instead

### Implementing Channels with STM

The `STM` version of `Chan` is called `TChan`:

```Haskell

data TChan a = TChan (TVar (TVarList a))
                     (TVar (TVarList a))

type TVarList a = TVar (TList a)
data TList a = TNil
             | TCons a (TVarList a)

newTChan :: STM (TChan a)
newTChan = do
    hole <- newTVar TNil
    read <- newTVar hole
    write <- newTVar hole
    return (TChan read write)

readTChan :: TChan a -> STM a
readTChan (TChan readVar _) = do
    listHead <- readTVar readVar
    head <- readTVar listHead
    case head of
        TNil -> retry
        TCons val tail -> do
            writeTVar readVar tail
            return val

writeTChan :: TChan a -> a -> STM ()
writeTChan (TChan _ writeVar) a = do
    newListEnd <- newTVar TNil
    listEnd <- readTVar writeVar
    writeTVar writeVar newListEnd
    writeTVar listEnd (TCons a newListEnd)
```

All the operations are in the STM monad, so they need to be wrapped in `atomically`
but they can all be composed.

The `TList` needs a `TNil` constructor to indicate an empty list;
in the `MVar` implementation, the empty list was just an empty `MVar`.

Blocking in `readTChan` is implemented by a call to `retry`.

We didn't have to worry anywhere about what happens when a read executes
concurrently with a write, because all operations are atomic.

#### More Operations are Possible

We can implement more operations thanks to STM:

```Haskell
unGetTChan :: TChan a -> a -> STM ()
unGetTChan (TChan readVar _) a = do
    listHead <- readTVar readVar
    newHead <- newTVar (TCons a listHead)
    writeTVar readVar newHead

isEmptyTChan :: TChan a -> STM Bool
isEmptyTChan (TChan read _) = do
    listhead <- readTVar read
    head <- readTVar listhead
    case head of
        TNil -> return True
        TCons _ _ -> return False
```

#### Composition of Blocking Operations

Since operations in STM can be composed togethre, we can build composite operations
like `readEitherTChan`:

```Haskell
readEitherTChan :: TChan a -> TChan b -> STM (Either a b)
readEitherTChan a b =
    fmap Left (readTChan a)
        `orElse`
    fmap Right (readTChan b)
```

#### Asynchronous Exception Safety

`STM` supports exceptions much like the `IO` monad, with two operations:

```Haskell
throwSTM :: Exception e => e -> STM a
catchSTM :: Exception e => STM a -> (e -> STM a) -> STM a
```

`throwSTM` throws an exception, and `catchSTM` catches exceptions and invokes
a handler.

In `catchSTM m h` if `m` raises an exception, then *all of its effects are
discarded*, and the nthe handler `h` is invoked.
If there is no enclosing `catchSTM` at all, then all of the transactions
effects are discarded and the exception is propogated out of `atomically`.

Here's an example to understand the motivatoin for this behavior:

```Haskell
readCheck :: TChan a -> STM a
readCheck chan = do
    a <- readTChan chan
    checkValue a
```

Where `checkValue` is an operatoins that imposes some constraint on the value read.
Now suppose `checkValue` raises an exception, we would prefer if the
`readTChan` had not happened because an element of the channel would be lost.
Furthermore, we would like `readCheck` to have this behavior regardless of
if there's an enclosing exception handler or not. Hence, `catchSTM` discards
the effects of its first argument if there's an exception.

The discarding-effects is very useful for asynchronous exceptions.
In most cases, asynchronous exception safety in the STM consists of doing
*absolutely nothing at all*. There are no locks to replace, no need for handlers,
no need for `bracket` or worrying about which critical sections to `mask`.

### An Alternative Channel Implementation

The flexibility of `STM` gives us more choices in how to construct channels,
we don't have to implement the same way as we did with `MVar`.

An `STM` operation can block on any condition whatsoever, so we can
represent channel contents in any data structure we want, even a simple list:

```Haskell
newtype TList a = TList (TVar [a])

newTList :: STM (TList a)
newTList = do
    v <- newTVar []
    return (TList v)

writeTList :: TList a -> a -> STM ()
writeTList (TList v) a = do
    list <- readTVar v
    writeTVar v (list ++ [a])

readTList :: TList a -> STM a
readTList (TList v) = do
    xs <- readTVar v
    case xs of
        [] -> retry
        (x:xs') -> do
            writeTVar v xs'
            return x
```

This abstraction has the exact same behavior as `TChan`.
The problem with this representation, though, is that Haskell uses linked lists
as its list datatype, so adding to the *end* of the list is an *O(n)* operation.
We can just create a queue data structure with *O(1)* enqueue and dequeue:

```Haskell
data TQueue a = TQueue (TVar [a]) (TVar [a])

newTQueue :: STM (TQueue a)
newTQueue = do
    read <- newTVar []
    write <- newTVar []
    return (TQueue read write)

writeTQueue :: TQueue a -> a -> STM ()
writeTQueue (TQueue _read write) a = do
    listend <- readTVar write
    writeTVar write (a:listend)

readTQueue :: TQueue a -> STM a
readTQueue (TQueue read write) = do
    xs <- readTVar read
    case xs of
        (x:xs') -> do writeTVar read xs'
                      return x
        [] -> do ys <- readTVar write
                 case ys of
                    [] -> retry
                    _ -> do let (z:zs) = reverse ys
                            writeTVar write []
                            writeTVar read zs
                            return z
```

*Note: for more details on these kinds of data structure,
read Okasaki's Purely Functional Data Structures (1999)*

Note the importance of the`reverse` being executed lazily with a let rather than
a case pattern match. We want it to be lazy so that the STM transaction
completes without having to do the `reverse`.

### Bounded Channels

Unbounded channels (`Chan` and `TChan`) can grow without bound if the reading
threads dont keep up with the writing threads, creating space issues.

Bounded channels (`MVar` and `TVar`) are limited by concurrency - if there
is a burst of writing activity, the writers will block waiting for the readers
to catch up.

A bounded channel is in between these two implementations with a limit on the size,
absorbing burst of writing activity without risking using too much memory.

Let's implement one with `STM`:

```Haskell
data TBQueue = TBQueue (TVar int) (TVar [a]) (TVar [a])

newTBQueue :: Int -> STM (TBQueue a)
newTBQueue size = do
    read <- newTVar []
    write <- newTVar []
    cap <- newTVar size
    return (TBQueue cap read write)

writeTBQueue :: TBQueue a -> a -> STM ()
writeTBQueue (TBQueue cap _read write) a = do
    avail <- readTVar cap
    if avail == 0
        then retry
        else writeTVar cap (avail -1)
    listend <- readTVar write
    writeTVar write (a:listend)

readTBQueue :: TBQueue a -> STM a
readTBQueue (TBQueue cap read write) a = do
    avail <- readTVar cap
    writeTVar cap (avail + 1)
    xs <- readTVar read
    case xs of
        (x:xs') -> do writeTVar read xs'
                      return x
        [] -> do ys <- readTVar write
                 case ys of
                    [] -> retry
                    _ -> do let (z:zs) = reverse ys
                            writeTVar write []
                            writeTVar read zs
                            return z
```

### What Can We Not Do with STM?

`MVar` is faster than STM, but we should not always assume using `MVar` will result
in faster code. For example `TList` (`STM`) outperforms `Chan` (`MVar`) and
`TList` has the advantage of being composable.

`MVar` also has the advantage of *fairness*: when multiple threads block on an `MVar`,
they are guaranteed to be woken up in FIFO order, and no single thread can be blocked
in `takeMVar` indefinitely.

In constrast, when multiple threads are blocked in STM transactions that depend
on a particular `TVar`, and the `TVar` is modified by another thread, it is not
enough to just wake up one of the blocked transactions - the runtime wakes them all.
Since a transaction can block on any arbitrary condition, the runtime doesn't know
what to wake up and in what order.

We can't implement fairness without sacrificing composability.

### Performance

If we can understand the cost of STM, we can avoid code that hits the bad cases.

An `STM` transaction works by keeping a *log* of `readTVar` and `writeTVar`
operations that have happened so far during the transaction. The log is used 3 ways:

1. Storing `writeTVar` operations in the log rather than immediately applying
   them in memory makes discarding the effects easy. So absorbing a
   transaction has a fixed small cost.
2. Each `readTVar` must traverse the log to check whether the `TVar` was
   written by an earlier `writeTVar`.
   Hence, `readTVar` is an *O(n)* operation in the length of the log.
3. The log contains a record of all the `readTVar` operations, so
   it can be used to discover the full set of `TVar`s read during the transaction,
   which we need to know in order to implement `retry`.

If the log matches the contents of memory at the end of an `STM` transaction,
the effects are *committed*, if not the log is discarded and we retry.

`STM` locks all `TVar`s involved in the transaction, but does not use global locks.
So two disjoint transactions can occur at the same time.

There are two important rules of thumb:

1. Never read an unbounded number of `TVar`s because the *O(n)* performance
   of `readTVar` gives *O(n^2)* for the whole transaction.
2. Try to avoid expensive evaluatoin inside a transaction because this will
   make the transaction take a long time, increasing the chance that another
   transaction will modify one or more of the same `TVar`s, causing the
   current transaction to be discarded and re-executed. In the worst case,
   a long transaction re-executes indefinitely because it is repeatedly
   aborted by shorter transactions.

`retry` uses the log to find out which `TVar`s were accessed by the transaction,
so it can trigger a rerun if any of them are changed.
So each `TVar` has a 'watch list' of threads that should be woken up if it's modified.
`retry` adds its thread to the `TVar`'s 'watch list'.
Hence, `retry` is *O(n)* in the number of `TVar`s read during the transaction.

Another thing to be careful about is composing too many blocking operations together.
For example, if we wanted to wait for a list of `TMVar`s to become full,
we might want to do this:

```Haskell
atomically $ mapM takeMVar ts
```

This is *O(n^2)* since we do the *O(n)* `takeMVar` *n* times (for each element)
after each `retry`. On the other hand this implementation is *O(n)*:

```Haskell
mapM (atomically . takeMVar) ts
```

and will be much faster.

### Summary

Benefits of STM:

1. **Composable atomicity**:
   able to construct arbitrarily large atomic operations on shared state,
   simplifying implementations of concurrent data structures with fine-grained
   locking.
2. **Composable blocking**:
   able to build operatoins that choose between multiple blocking operations,
   which is very difficult with `MVar`s and other low-level concurrency abstractions.
3. **Robustness in the presence of failure and cancellation**:
   a transaction in progress is aborted if an exception occurs,
   so `STM` makes it easy to maintain invariants on state in the presence of
   exceptions.
