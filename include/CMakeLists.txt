# Install public headers

# Install ttlang headers
install(DIRECTORY ttlang/
  DESTINATION include/ttlang
  COMPONENT TTLangHeaders
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
)

# Install ttlang-c headers (C bindings)
install(DIRECTORY ttlang-c/
  DESTINATION include/ttlang-c
  COMPONENT TTLangHeaders
  FILES_MATCHING
  PATTERN "*.h"
)
