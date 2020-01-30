((c++-mode . (
              ;; project-dependent include paths
              (eval .
                    (set (make-local-variable 'irony-additional-clang-options)
                         (let ((project-path '"/home/sehrig/programming/alpaka/hello_world"))
                           (progn
                             (setq irony-additional-clang-options
                                   (append
                                    (list "-I" "/home/sehrig/tmp/alpaka/include")
                                    irony-additional-clang-options))
                             ));; end let
                         ) ;; end set
                    ) ;; end eval
              ;; Increasing the timeout, because compiling some files for auto-completion takes a long time
              (company-async-timeout . 10)
              )
           )) ;; end c++-mode
