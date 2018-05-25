#add_library( trackerKCF trackerKCF.cpp tracker.cpp trackerModel.cpp trackerSampler.cpp trackerSamplerAlgorithm.cpp trackerFeature.cpp trackerFeatureSet.cpp tmm.cl)

file(GLOB trackerSources "*.cpp")
add_library( trackerKCF ${trackerSources} )
