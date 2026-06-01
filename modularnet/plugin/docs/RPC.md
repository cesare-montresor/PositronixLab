
                    # Dispatch manager commands
                    if command == "get_env_variable":
                        key, default = args[0], args[1] if len(args) > 1 else None
                        result = self.env_set(key, default)
                    
                    elif command == "set_env_variable":
                        key, value = args[0], args[1]
                        self.env_get(key, value)
                        result = None
                    
                    elif command == "get_all_env_variables":
                        result = self.env_get_all()
                    
                    elif command == "publish":
                        topic, data = args[0], args[1]
                        result = self.publish(topic, data)
                    
                    elif command == "call_service":
                        target_plugin_id, method_name = args[0], args[1]
                        service_args = args[2:]
                        result = self.call_service(target_plugin_id, method_name, *service_args, **kwargs)
                    
                    elif command == "get_plugin_status":
                        target_plugin_id = args[0]
                        result = self.get_plugin_status(target_plugin_id)
                        if result:
                            result = result.__dict__  # Convert ProcessStatus to dict
                    
                    elif command == "get_all_plugin_status":
                        status_dict = self.get_all_plugin_status()
                        result = {pid: s.__dict__ for pid, s in status_dict.items()}
                    
                    else: