#include <iostream>
#include <math.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <limits>
#include <dbconnector/dbconnector.hpp>
#include <modules/shared/HandleTraits.hpp>
#include "gmm.h"

namespace madlib {

  // Use Eigen
  using namespace dbal::eigen_integration;

  namespace modules {

    // Import names from other MADlib modules
    using dbal::NoSolutionFoundException;

    namespace gmm {

      /**
       * class GMMUpdateWeightMeanTransitionState stores
       * the intermediate state of the UDA gmm_update_weight.
       * To database the state is a double precison[], but we
       * cast/rebind it to an object here.
       */
      template <class Handle>
      class GMMUpdateWeightMeanTransitionState {
        template <class OtherHandle>
        friend class GMMUpdateWeightMeanTransitionState;

        public:
          GMMUpdateWeightMeanTransitionState(const AnyType &inArray)
          : mStorage( inArray.getAs<Handle>() ) {

            rebind(static_cast<uint16_t>(mStorage[1]));
          }

          inline operator AnyType() const {
            return mStorage;
          }

          inline void initialize( const Allocator &inAllocator, uint16_t inWidthOfX ) {
            mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
            dbal::DoZero, dbal::ThrowBadAlloc>( arraySize(inWidthOfX) );

            rebind(inWidthOfX);
            widthOfX = inWidthOfX;
          }

          template <class OtherHandle>
          GMMUpdateWeightMeanTransitionState &operator= (
            const GMMUpdateWeightMeanTransitionState<OtherHandle> &inOtherState ) {

            for(size_t i = 0; i < mStorage.size(); ++i ) {
              mStorage[i]  = inOtherState.mStorage[i];
            }

            return *this;
          }

          template <class OtherHandle>
          GMMUpdateWeightMeanTransitionState &operator += (
            const GMMUpdateWeightMeanTransitionState<OtherHandle> &inOtherState ) {
                if( mStorage.size() != inOtherState.mStorage.size() ||
                  widthOfX != inOtherState.widthOfX ) {
                  throw std::logic_error("Internal error: Incompatible transition "
                    "states");
                }

                numRows += inOtherState.numRows;
                weight_vec += inOtherState.weight_vec;
                mean_mat += inOtherState.mean_mat;

                return *this;
            }

          inline void reset() {
            numRows = 0;
          }

          private:
            static inline size_t arraySize( const uint16_t inWidthOfX ) {
              return 3 + inWidthOfX + inWidthOfX * inWidthOfX;
            }

            /**
            * @brief Rebind to a new storage array
            *
            * @param inWidthOfX The number of independent variables.
            *
            * Inter-iteration components ( updated in transition step ):
            * - 0: iteration ( current iteration )
            * - 1: widthOfX ( dimensionality of input )
            * - 2: numRows ( number of rows already processed in this iteration )
            *
            *
            * Intra-iteration components ( updated in transition step ) :
            * - 3 + widthOfX : weight_vec
            * - 3 + widthOfX + widthOfX * widthOfX : mean_mat
            */

            void rebind( uint16_t inWidthOfX ) {
              iteration.rebind(&mStorage[0]);
              widthOfX.rebind(&mStorage[1]);
              numRows.rebind(&mStorage[2]);
              weight_vec.rebind(&mStorage[3], inWidthOfX);
              mean_mat.rebind(&mStorage[3 + inWidthOfX], inWidthOfX, inWidthOfX);
            }

            Handle mStorage;

            public:
              typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
              typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
              typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
              typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap weight_vec;
              typename HandleTraits<Handle>::MatrixTransparentHandleMap mean_mat;
          };

          template <class Handle>
          class GMMUpdateCovarTransitionState {
            template <class OtherHandle>
            friend class GMMUpdateCovarTransitionState;

            public:
              GMMUpdateCovarTransitionState( const AnyType & inArray )
              : mStorage(inArray.getAs<Handle>()) {

                rebind(static_cast<uint16_t>(mStorage[1]));
              }

              inline operator AnyType() const {
                return mStorage;
              }

              inline void initialize( const Allocator &inAllocator, uint16_t inWidthOfX ) {
                mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));

                rebind(inWidthOfX);
                widthOfX = inWidthOfX;
              }

              template <class OtherHandle>
              GMMUpdateCovarTransitionState & operator=(
                const GMMUpdateCovarTransitionState<OtherHandle> &inOtherState ) {

                for (size_t i = 0; i < mStorage.size(); ++i) {
                  mStorage[i] = inOtherState.mStorage[i];
                }

                return *this;
              }

              template <class OtherHandle>
              GMMUpdateCovarTransitionState &operator+=(
                const GMMUpdateCovarTransitionState<OtherHandle> &inOtherState) {

                if (mStorage.size() != inOtherState.mStorage.size() ||
                  widthOfX != inOtherState.widthOfX)
                  throw std::logic_error("Internal error: Incompatible transition "
                    "states");

                numRows += inOtherState.numRows;
                covar_mat += inOtherState.covar_mat;

                return *this;
              }

              inline void reset() {
                numRows = 0;
              }

              private:
                static inline size_t arraySize( const uint16_t inWidthOfX ) {
                  return 4 + inWidthOfX * inWidthOfX;
                }

              void rebind( uint16_t inWidthOfX ) {
                iteration.rebind(&mStorage[0]);
                widthOfX.rebind(&mStorage[1]);
                numClusters.rebind(&mStorage[2]);
                numRows.rebind(&mStorage[3]);
                covar_mat.rebind(&mStorage[4], inWidthOfX, inWidthOfX);
              }

              Handle mStorage;

              public:
                typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
                typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
                typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
                typename HandleTraits<Handle>::ReferenceToUInt32 numClusters;
                typename HandleTraits<Handle>::MatrixTransparentHandleMap covar_mat;
          };

          AnyType gmm_update_covariance_transition::run (AnyType & args) {
            GMMUpdateCovarTransitionState<MutableArrayHandle<double> > state = args[0];
            if( args[1].isNull() || args[2].isNull() || args[4].isNull() || args[5].isNull()) { return args[0]; }

            MappedColumnVector x;
            try {
              MappedColumnVector xx = args[1].getAs<MappedColumnVector>();
              x.rebind(xx.memoryHandle(), xx.size());
            } catch ( const ArrayWithNullException &e) {
              return args[0];
            }

            MappedColumnVector z;
            try {
              MappedColumnVector zz = args[2].getAs<MappedColumnVector>();
              z.rebind(zz.memoryHandle(), zz.size());
            } catch (const ArrayWithNullException &e ) {
              return args[0];
            }

            MappedColumnVector w;
            try {
              MappedColumnVector ww = args[3].getAs<MappedColumnVector>();
              w.rebind(ww.memoryHandle(), ww.size());
            } catch (const ArrayWithNullException &e ) {
              return args[0];
            }

            if( state.numRows == 0 ) {
              state.initialize(*this, static_cast<uint16_t>(x.size()));
              int numClusters = args[3].getAs<int>();
              state.numClusters = numClusters;
            }

            state.numRows++;

            MappedMatrix mean_mat = args[5].getAs<MappedMatrix>();

            for(int m = 0; m < mean_mat.rows(); ++m ) {
              ColumnVector cv = mean_mat.row(m).transpose();
              state.covar_mat += ( ( z(m) * (x - cv) * trans(x - cv) ) / w(m) );
            }

            return state;
          }

          AnyType gmm_update_covariance_merge::run ( AnyType & args) {
            GMMUpdateCovarTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
            GMMUpdateCovarTransitionState<ArrayHandle<double> > stateRight = args[1];

            if(stateLeft.numRows == 0) {
              return stateRight;
            } else if( stateRight.numRows == 0) {
              return stateLeft;
            }

            stateLeft += stateRight;
            return stateLeft;
          }

          AnyType gmm_update_covariance_final::run ( AnyType & args) {

            GMMUpdateCovarTransitionState<MutableArrayHandle<double> > state = args[0];
            state.covar_mat /= static_cast<double>(state.numRows * state.numClusters);
            state.iteration++;
            return state;
          }

          AnyType gmm_update_weight_transition::run( AnyType &args ) {
            GMMUpdateWeightMeanTransitionState<MutableArrayHandle<double> > state = args[0];
            if( args[1].isNull() || args[2].isNull() ) { return args[0]; }
            MappedColumnVector x;
            try {

                  // an exception is raised in the backend if args[1] contains nulls
              MappedColumnVector xx = args[1].getAs<MappedColumnVector>();

                  // x is a const reference, we can only rebind to change its pointer
              x.rebind(xx.memoryHandle(), xx.size());
            } catch (const  ArrayWithNullException &e) {
              return args[0];
            }

            MappedColumnVector z;
            try {
              MappedColumnVector zz = args[2].getAs<MappedColumnVector>();
              z.rebind(zz.memoryHandle(), zz.size());
            } catch (const ArrayWithNullException &e ) {
              return args[0];
            }

            if(state.numRows == 0) {
              state.initialize(*this, static_cast<uint16_t>(x.size()));
            }

            state.numRows++;
            state.weight_vec += z;
            state.mean_mat += z * trans(x);

            return state;
          }

          AnyType gmm_update_weight_merge::run ( AnyType & args ) {
            GMMUpdateWeightMeanTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
            GMMUpdateWeightMeanTransitionState<ArrayHandle<double> > stateRight = args[1];

            if(stateLeft.numRows == 0) {
              return stateRight;
            } else if( stateRight.numRows == 0) {
              return stateLeft;
            }

            stateLeft += stateRight;
            return stateLeft;
          }

          AnyType gmm_update_weight_final::run ( AnyType & args ) {

            GMMUpdateWeightMeanTransitionState<MutableArrayHandle<double> > state = args[0];

            for(int i = 0; i < state.widthOfX; ++i ) {
              double weight = static_cast<double>(state.weight_vec(i));
              for(int j = 0; j < state.widthOfX; ++j ) {
                state.mean_mat(i,j) = state.mean_mat(i,j) / weight;
              }
            }

            state.weight_vec /= static_cast<double> (state.numRows);

            AnyType pair;
            pair << state.weight_vec << state.mean_mat;

            return pair;
          }

          AnyType gmm_e_single_step::run( AnyType & args ) {
            MappedColumnVector data_vec = args[0].getAs<MappedColumnVector>();
            MappedColumnVector weight_vec = args[1].getAs<MappedColumnVector>();
            MappedMatrix mean_mat = args[2].getAs<MappedMatrix>();
            MappedMatrix covar_mat = args[3].getAs<MappedMatrix>();

            Matrix covar_mat_inv = covar_mat.inverse();

                // @todo check if covar_mat_inv is not finite

            MutableNativeColumnVector cluster_mem_vec(
              this->allocateArray<double>(weight_vec.size()) );

            double sum = 0.0;
            for( int i = 0; i < cluster_mem_vec.size(); ++i ) {
              RowVector mean_row = mean_mat.row(i);
              ColumnVector mean_vec = mean_row.transpose();
              double term_in_exp = static_cast<double> (trans(data_vec - mean_vec) * covar_mat_inv * (data_vec - mean_vec));
              cluster_mem_vec(i) = static_cast<double> (std::exp(-0.5 * term_in_exp * weight_vec(i)));

              sum += cluster_mem_vec(i);
            }

            cluster_mem_vec /= sum;

            return cluster_mem_vec;
          }

          AnyType gmm_compute_log_likelihood_transition::run (AnyType & args ) {
            double log_likelihood = args[0].getAs<double>();
            if( args[1].isNull() || args[2].isNull() || args[3].isNull() || args[4].isNull() || args[5].isNull() ) {
              return args[0];
            }

            // data point
            MappedColumnVector x;
            try {
              MappedColumnVector xx = args[1].getAs<MappedColumnVector>();
              x.rebind(xx.memoryHandle(), xx.size());
            } catch (const ArrayWithNullException &e) {
              return args[0];
            }

            // cluster membership
            MappedColumnVector z;
            try {
              MappedColumnVector zz = args[2].getAs<MappedColumnVector>();
              z.rebind(zz.memoryHandle(), zz.size());
            } catch (const ArrayWithNullException &e ) {
              return args[0];
            }

            // weight vector
            MappedColumnVector w;
            try {
              MappedColumnVector ww = args[3].getAs<MappedColumnVector>();
              w.rebind(ww.memoryHandle(), ww.size());
            } catch (const ArrayWithNullException &e ) {
              return args[0];
            }

            // Mean matrix and covariance matrix
            MappedMatrix mean_mat = args[4].getAs<MappedMatrix>();
            MappedMatrix covar_mat = args[5].getAs<MappedMatrix>();

            Matrix covar_mat_inv = covar_mat.inverse();

            long int d = x.size();
            double det = covar_mat.determinant();
            double const_term = -1 * ((double)d / 2.0) * std::log(2*M_PI*det);
            for( int i = 0; i < z.size(); ++i ) {
              double temp = 0.0;
              RowVector mean_row = mean_mat.row(i);
              ColumnVector mean_vec = mean_row.transpose();
              double term_in_exp = static_cast<double>(trans(x - mean_vec) * covar_mat_inv * (x - mean_vec));
              temp = const_term + (-0.5*term_in_exp);
              temp += std::log(w(i));
              temp *= z(i);
              log_likelihood += temp;
            }

            return log_likelihood;
          }

          AnyType gmm_compute_log_likelihood_merge::run (AnyType & args ) {
            double log_lk_left = args[0].getAs<double>();
            double log_lk_right = args[1].getAs<double>();
            double log_lk = log_lk_left + log_lk_right;
            return log_lk;
          }
        }
      }
    }
