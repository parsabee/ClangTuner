//
// Created by Parsa Bagheri on 8/25/21.
//

#ifndef TUNER__LOCKABLEWRAPPER_HPP
#define TUNER__LOCKABLEWRAPPER_HPP

#include <future>
#include <list>
#include <mutex>
#include <thread>

template <typename T> using FuturePtr = std::future<std::unique_ptr<T>>;

// Forward ref
template <typename T, typename Mutex, typename... Args> class Lockable;

/// A Locked object acquires the lock of a Lockable object upon construction
/// and releases the lock upon destruction. After locking, caller can access
/// the object within Lockable object by using getObject()
template <typename T, typename Mutex> class Locked {
  friend class Lockable<T, Mutex>;
  std::lock_guard<Mutex> lock;

protected:
  Lockable<T, Mutex> &lockableObj;

public:
  explicit Locked(Lockable<T, Mutex> &obj) : lock(obj.mtx), lockableObj(obj) {}
  Locked(const Locked<T, Mutex> &) = delete;
  Locked(Locked<T, Mutex> &&) = delete;
  Locked<T, Mutex> operator=(const Locked<T, Mutex> &) = delete;
  ~Locked() = default;
  T &getObject() { return lockableObj.Obj; }
};

/// A LockableObject has a mutex that can be locked by passing it to Locked
/// constructor. The only way to access the object is by
/// locking it first
template <typename T, typename Mutex, typename... Args> class Lockable {
  friend class Locked<T, Mutex>;

protected:
  // The object
  T Obj;

  // The lock
  Mutex mtx;

public:
  Lockable(T object) : Obj(std::move(object)) {}
  Lockable(Args &&... args) : Obj(std::forward(args)...) {}
  Lockable(const Lockable<T, Mutex> &other) = delete;
  Lockable(Lockable<T, Mutex> &&other)
      : Obj(std::move(other.Obj)), mtx(std::move(other.mtx)) {}
  Locked<T, Mutex> lock() { return Locked<T, Mutex>(*this); }
};

template <typename T, typename Mutex> class QueueProducer;

template <typename T, typename Mutex> class QueueConsumer;

template <typename T, typename Mutex>
class LockableQueue : public Lockable<std::list<T>, Mutex> {
  friend class QueueProducer<T, Mutex>;
  friend class QueueConsumer<T, Mutex>;

protected:
  bool isClosed = false; // The producer of the list can close it

  // Needed to synchronize enqueue and dequeue operations
  std::condition_variable cnd;

  std::list<T> &getList() { return this->Obj; }

public:
  LockableQueue()
      : Lockable<std::list<T>, Mutex>() {}
};

template <typename T, typename Mutex> class QueueProducer {
  LockableQueue<T, Mutex> &lockableQueue;

public:
  QueueProducer(LockableQueue<T, Mutex> &lockableQueue)
      : lockableQueue(lockableQueue) {}

  QueueProducer(const QueueProducer &) = delete;

  /// Closes the list (enqueue operations will fail).
  /// After closing, it cannot be reopened
  void close() {
    std::lock_guard lg(lockableQueue.mtx);
    lockableQueue.isClosed = true;
    lockableQueue.cnd.notify_all(); // let all of the threads know that it's
                                    // closed
  }

  bool lockAndEnqueue(T future) {
    std::lock_guard lg(lockableQueue.mtx);
    if (lockableQueue.isClosed)
      return false;

    lockableQueue.getList().push_back(std::move(future));
    lockableQueue.cnd.notify_one(); // let only one thread know that one item
                                    // is pushed to the queue
    return true;
  }
};

template <typename T, typename Mutex> class QueueConsumer {
  LockableQueue<T, Mutex> &lockableQueue;

public:
  QueueConsumer(LockableQueue<T, Mutex> &lockableQueue)
      : lockableQueue(lockableQueue) {}

  QueueConsumer(const QueueConsumer &) = delete;

  bool lockAndDequeue(T &future) {
    std::lock_guard lg(lockableQueue.mtx);
    if (lockableQueue.getList().empty())
      return false;

    future = std::move(lockableQueue.Obj.front());
    lockableQueue.getList().pop_front();
    return true;
  }

  /// Waits until either there are elements to be processed in the list
  /// or the predicate is satisfied
  void waitUntilAvailableOr(std::function<bool(void)> predicate) {
    std::unique_lock<Mutex> ul(lockableQueue.mtx);
    lockableQueue.cnd.wait(ul, [this, &predicate] {
      return predicate() || !lockableQueue.getList().empty();
    });
  }
};

#endif // TUNER__LOCKABLEWRAPPER_HPP
