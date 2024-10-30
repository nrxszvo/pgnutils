add_library(PgnParser parseMoves.cpp inference.cpp validate.cpp)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	target_include_directories(PgnParser PUBLIC /opt/homebrew/include)
endif()
