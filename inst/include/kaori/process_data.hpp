#ifndef KAORI_PROCESS_DATA_HPP
#define KAORI_PROCESS_DATA_HPP

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <cstddef>

#include "FastqReader.hpp"

#include "byteme/byteme.hpp"

/**
 * @file process_data.hpp
 *
 * @brief Process single- or paired-end data.
 */

namespace kaori {

/**
 * @cond
 */
typedef std::size_t ReadIndex;

class ChunkOfReads {
public:
    ChunkOfReads() : my_sequence_offset(1), my_name_offset(1) {} // zero is always the first element.

    void clear(bool use_names) {
        my_sequence_buffer.clear();
        my_sequence_offset.resize(1);
        if (use_names) {
            my_name_buffer.clear();
            my_name_offset.resize(1);
        }
    }

    void add_read_sequence(const std::vector<char>& sequence) {
        add_read_details(sequence, my_sequence_buffer, my_sequence_offset);
    }

    void add_read_name(const std::vector<char>& name) {
        add_read_details(name, my_name_buffer, my_name_offset);
    }

    ReadIndex size() const {
        return my_sequence_offset.size() - 1;
    }

    std::pair<const char*, const char*> get_sequence(ReadIndex i) const {
        return get_details(i, my_sequence_buffer, my_sequence_offset);
    }

    std::pair<const char*, const char*> get_name(ReadIndex i) const {
        return get_details(i, my_name_buffer, my_name_offset);
    }

private:
    std::vector<char> my_sequence_buffer;
    std::vector<std::size_t> my_sequence_offset;
    std::vector<char> my_name_buffer;
    std::vector<std::size_t> my_name_offset;

    static void add_read_details(const std::vector<char>& src, std::vector<char>& dst, std::vector<std::size_t>& offset) {
        dst.insert(dst.end(), src.begin(), src.end());
        auto last = offset.back();
        offset.push_back(last + src.size());
    }

    static std::pair<const char*, const char*> get_details(ReadIndex i, const std::vector<char>& dest, const std::vector<std::size_t>& offset) {
        const char * base = dest.data();
        return std::make_pair(base + offset[i], base + offset[i + 1]);
    }
};

template<typename Workspace_>
class ThreadPool {
public:
    template<typename RunJob_>
    ThreadPool(RunJob_ run_job, int num_threads) : my_helpers(num_threads) {
        std::mutex init_mut;
        std::condition_variable init_cv;
        int num_initialized = 0;

        my_threads.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            // Copy lambda as it will be gone once this constructor finishes.
            my_threads.emplace_back([run_job,this,&init_mut,&init_cv,&num_initialized](int thread) -> void { 
                Helper env; // allocating this locally within each thread to reduce the risk of false sharing.
                my_helpers[thread] = &env;
                {
                    std::lock_guard lck(init_mut);
                    ++num_initialized;
                    init_cv.notify_one();
                }

                while (1) {
                    std::unique_lock lck(env.mut);
                    env.cv.wait(lck, [&]() -> bool { return env.input_ready; });
                    if (env.terminated) {
                        return;
                    }
                    env.input_ready = false;

                    try {
                        run_job(env.work);
                    } catch (...) {
                        std::lock_guard elck(my_error_mut);
                        if (!my_error) {
                            my_error = std::current_exception();
                        }
                    }

                    env.has_output = true;
                    env.available = true;
                    env.cv.notify_one();
                }
            }, t);
        }

        // Only returning once all threads (and their specific mutexes) are initialized.
        {
            std::unique_lock ilck(init_mut);
            init_cv.wait(ilck, [&]() -> bool { return num_initialized == num_threads; });
        }
    }

    ~ThreadPool() {
        for (auto envptr : my_helpers) {
            auto& env = *envptr;
            {
                std::lock_guard lck(env.mut);
                env.terminated = true;
                env.input_ready = true;
            }
            env.cv.notify_one();
        }
        for (auto& thread : my_threads) {
            thread.join();
        }
    }

private:
    std::vector<std::thread> my_threads;

    struct Helper {
        std::mutex mut;
        std::condition_variable cv;
        bool input_ready = false;
        bool available = true;
        bool has_output = false;
        bool terminated = false;
        Workspace_ work;
    };
    std::vector<Helper*> my_helpers;

    std::mutex my_error_mut;
    std::exception_ptr my_error;

public:
    template<typename CreateJob_, typename MergeJob_>
    void run(CreateJob_ create_job, MergeJob_ merge_job) {
        auto num_threads = my_threads.size();
        bool finished = false;
        decltype(num_threads) thread = 0, finished_count = 0;

        // We submit jobs by cycling through all threads, then we merge their results in order of submission.
        // This is a less efficient worksharing scheme but it guarantees the same order of merges.
        while (1) {
            auto& env = (*my_helpers[thread]);
            std::unique_lock lck(env.mut);
            env.cv.wait(lck, [&]() -> bool { return env.available; });

            {
                std::lock_guard elck(my_error_mut);
                if (my_error) {
                    std::rethrow_exception(my_error);
                }
            }
            env.available = false;

            if (env.has_output) {
                merge_job(env.work);
                env.has_output = false;
            }

            if (finished) {
                // Go through all threads one last time, making sure all results are merged.
                ++finished_count;
                if (finished_count == num_threads) {
                    break;
                }
            } else {
                // 'create_job' is responsible for parsing the FASTQ file and
                // creating a ChunkOfReads. Unfortunately I can't figure out
                // how to parallelize the parsing as it is very hard to jump
                // into a middle of a FASTQ file and figure out where the next
                // entry is; this is because (i) multiline sequences are legal
                // and (ii) +/@ are not sufficient delimiters when they can
                // show up in the quality scores.
                finished = create_job(env.work);
                env.input_ready = true;
                lck.unlock();
                env.cv.notify_one();
            }

            ++thread;
            if (thread == num_threads) {
                thread = 0;
            }
        }
    }
};

/**
 * @endcond
 */

/**
 * @brief Options for `process_single_end_data()`.
 */
struct ProcessSingleEndDataOptions {
    /**
     * Number of threads to use for processing reads after FASTQ parsing.
     */
    int num_threads = 1;

    /**
     * Size of the buffer for storing bytes from a FASTQ file prior to parsing.
     * Larger values improve speed at the cost of increased memory usage.
     */
    std::size_t buffer_size = 65535;

    /**
     * Number of reads to be processed in each thread.
     * Smaller values improve work-sharing granularity but increase multi-threading overhead. 
     */
    std::size_t block_size = 65535; // use the smallest maximum value for a size_t.
};

/**
 * Run a handler for each read in single-end data, by calling `handler.process()` on each read.
 * It is expected that the results are stored in `handler` for retrieval by the caller.
 *
 * @tparam Pointer_ Pointer to a class that serves as a source of input bytes.
 * The pointed-to class should satisfy the `byteme::Reader` interface; it may also be a concrete `byteme::Reader` subclass to enable devirtualization. 
 * Either a smart or raw pointer may be supplied depending on how the caller wants to manage the lifetime of the pointed-to object. 
 * @tparam Handler_ Class that implements a handler for single-end data.
 *
 * @param input Pointer to an input byte source containing data from a single-end FASTQ file.
 * @param handler Handler instance for single-end data. 
 * @param options Further options.
 *
 * @section single-handler-req Single-end handler requirements
 * The `Handler_` class is expected to implement the following methods:
 * - `initialize()`: this should be a thread-safe `const` method that returns a state object (denoted here as having type `State`, though the exact name may vary).
 *   The idea is to store results in the state object for thread-safe execution.
 *   The state object should be default-constructible.
 * - `reduce(State& state)`: this should merge the results from the `state` object into the `Handler` instance.
 *   This will be called in a serial section and does not have to be thread-safe.
 *
 * The `Handler` should have a static `constexpr` variable `use_names`, indicating whether or not names should be passed to the `process()` method.
 *
 * If `use_names` is `false`, the `Handler` class should implement:
 * - `process(State& state, const std::pair<const char*, const char*>& seq)`: this should be a `const` method that processes the read in `seq` and stores its results in `state`.
 *   `seq` will contain pointers to the start and one-past-the-end of the read sequence.
 *
 * Otherwise, if `use_names` is `true`, the class should implement:
 * - `process(State& state, const std::pair<const char*, const char*>& name, const std::pair<const char*, const char*>& seq)`: 
 *    this should be a `const` method that processes the read in `seq` and stores its results in `state`.
 *   `name` will contain pointers to the start and one-past-the-end of the read name.
 *   `seq` will contain pointers to the start and one-past-the-end of the read sequence.
 */
template<typename Pointer_, class Handler_>
void process_single_end_data(Pointer_ input, Handler_& handler, const ProcessSingleEndDataOptions& options) {
    struct SingleEndWorkspace {
        ChunkOfReads reads;
        decltype(handler.initialize()) state;
    };

    FastqReader<Pointer_> fastq(input, options.buffer_size);
    const Handler_& conhandler = handler; // Safety measure to enforce const-ness within each thread.

    ThreadPool<SingleEndWorkspace> tp(
        [&](SingleEndWorkspace& work) -> void {
            auto& state = work.state;
            state = conhandler.initialize(); // reinitializing for simplicity and to avoid accumulation of reserved memory.
            const auto& curreads = work.reads;
            auto nreads = curreads.size();

            if constexpr(!Handler_::use_names) {
                for (decltype(nreads) b = 0; b < nreads; ++b) {
                    conhandler.process(state, curreads.get_sequence(b));
                }
            } else {
                for (decltype(nreads) b = 0; b < nreads; ++b) {
                    conhandler.process(state, curreads.get_name(b), curreads.get_sequence(b));
                }
            }
        },
        options.num_threads
    );

    tp.run(
        [&](SingleEndWorkspace& work) -> bool {
            auto& curreads = work.reads;
            for (ReadIndex b = 0; b < options.block_size; ++b) {
                if (!fastq()) {
                    return true;
                }

                curreads.add_read_sequence(fastq.get_sequence());
                if constexpr(Handler_::use_names) {
                    curreads.add_read_name(fastq.get_name());
                }
            }
            return false;
        },
        [&](SingleEndWorkspace& work) -> void {
            handler.reduce(work.state);
            work.reads.clear(Handler_::use_names);
        }
    );
}

/**
 * @brief Options for `process_paired_end_data()`.
 */
struct ProcessPairedEndDataOptions {
    /**
     * Number of threads to use for processing after FASTQ parsing.
     */
    int num_threads = 1;

    /**
     * Size of the buffer for storing bytes from a FASTQ file prior to parsing.
     * Larger values improve speed at the cost of increased memory usage.
     */
    std::size_t buffer_size = 65535;

    /**
     * Number of reads to process in each thread.
     * Smaller values improve work-sharing granularity but increase multi-threading overhead. 
     */
    std::size_t block_size = 100000;
};

/**
 * Run a handler for each read in paired-end data, by calling `handler.process()` on each read pair.
 * It is expected that the results are stored in `handler` for retrieval by the caller.
 *
 * @tparam Pointer_ Pointer to a class that serves as a source of input bytes.
 * The pointed-to class should satisfy the `byteme::Reader` interface; it may also be a concrete `byteme::Reader` subclass to enable devirtualization. 
 * Either a smart or raw pointer may be supplied depending on how the caller wants to manage the lifetime of the pointed-to object. 
 * @tparam Handler_ A class that implements a handler for paired-end data.
 *
 * @param input1 Pointer to an input byte source containing data from one of the paired-end FASTQ files.
 * This may or may not need to be specifically "read 1", depending on the `handler`.
 * @param input2 Pointer to an input byte source containing data from the other paired-end FASTQ file.
 * This may or may not need to be specifically "read 2", depending on the `handler`.
 * @param handler Handler instance for paired-end data. 
 * @param options Further options.
 *
 * @section paired-handler-req Paired-end handler requirements
 * The `Handler` class is expected to implement the following methods:
 * - `initialize()`: this should be a thread-safe `const` method that returns a state object (denoted here as having type `State`, though the exact name may vary).
 *   The idea is to store results in the state object for thread-safe execution.
 *   The state object should be default-constructible.
 * - `reduce(State& state)`: this should merge the results from the `state` object into the `Handler` instance.
 *   This will be called in a serial section and does not have to be thread-safe.
 *
 * The `Handler` should have a static `constexpr` variable `use_names`, indicating whether or not names should be passed to the `process()` method.
 *
 * If `use_names` is `false`, the `Handler` class should implement:
 * - `process(State& state, const std::pair<const char*, const char*>& seq1, const std::pair<const char*, const char*>& seq2)`: 
 *   this should be a `const` method that processes the paired reads in `seq1` and `seq2`, and stores its results in `state`.
 *   `seq1` and `seq2` will contain pointers to the start and one-past-the-end of the sequences of the paired reads.
 *
 * Otherwise, if `use_names` is `true`, the class should implement:
 * - `process(State& state, const std::pair<const char*, const char*>& name1, const std::pair<const char*, const char*>& seq1, const std::pair<const char*, const char*>& name2, const std::pair<const char*, const char*>& seq2)`: 
 *   this should be a `const` method that processes the paired reads in `seq1` and `seq2`, and stores its results in `state`.
 *   `name1` and `name2` will contain pointers to the start and one-past-the-end of the read names.
 *   `seq1` and `seq2` will contain pointers to the start and one-past-the-end of the read sequences.
 */
template<class Pointer_, class Handler_>
void process_paired_end_data(Pointer_ input1, Pointer_ input2, Handler_& handler, const ProcessPairedEndDataOptions& options) {
    struct PairedEndWorkspace {
        ChunkOfReads reads1, reads2;
        decltype(handler.initialize()) state;
    };

    FastqReader<Pointer_> fastq1(input1, options.buffer_size);
    FastqReader<Pointer_> fastq2(input2, options.buffer_size);
    const Handler_& conhandler = handler; // Safety measure to enforce const-ness within each thread.

    ThreadPool<PairedEndWorkspace> tp(
        [&](PairedEndWorkspace& work) -> void {
            auto& state = work.state;
            state = conhandler.initialize(); // reinitializing for simplicity and to avoid accumulation of reserved memory.
            const auto& curreads1 = work.reads1;
            const auto& curreads2 = work.reads2;
            ReadIndex nreads = curreads1.size();

            if constexpr(!Handler_::use_names) {
                for (ReadIndex b = 0; b < nreads; ++b) {
                    conhandler.process(state, curreads1.get_sequence(b), curreads2.get_sequence(b));
                }
            } else {
                for (ReadIndex b = 0; b < nreads; ++b) {
                    conhandler.process(
                        state,
                        curreads1.get_name(b), 
                        curreads1.get_sequence(b),
                        curreads2.get_name(b), 
                        curreads2.get_sequence(b)
                    );
                }
            }
        },
        options.num_threads
    );

    tp.run(
        [&](PairedEndWorkspace& work) -> bool {
            bool finished1 = false;
            {
                auto& curreads = work.reads1;
                for (ReadIndex b = 0; b < options.block_size; ++b) {
                    if (!fastq1()) {
                        finished1 = true;
                        break;
                    }

                    curreads.add_read_sequence(fastq1.get_sequence());
                    if constexpr(Handler_::use_names) {
                        curreads.add_read_name(fastq1.get_name());
                    }
                }
            }

            bool finished2 = false;
            {
                auto& curreads = work.reads2;
                for (ReadIndex b = 0; b < options.block_size; ++b) {
                    if (!fastq2()) {
                        finished2 = true;
                        break;
                    }

                    curreads.add_read_sequence(fastq2.get_sequence());
                    if constexpr(Handler_::use_names) {
                        curreads.add_read_name(fastq2.get_name());
                    }
                }
            }

            if (finished1 != finished2 || work.reads1.size() != work.reads2.size()) {
                throw std::runtime_error("different number of reads in paired FASTQ files");
            }
            return finished1;
        },
        [&](PairedEndWorkspace& work) -> void {
            handler.reduce(work.state);
            work.reads1.clear(Handler_::use_names);
            work.reads2.clear(Handler_::use_names);
        }
    );
}

}

#endif
